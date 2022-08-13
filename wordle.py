#!/usr/bin/python3

import unicodedata
import json
from math import sqrt
import numpy as np
from tqdm import tqdm
from os.path import join, dirname
import sys

max_nb_chars = 10
base = [3**i for i in range(max_nb_chars+1)]
max_combinations_to_try = 2000000
min_words_to_try = 200
default_freq = 0.001

def import_csv_scrabble(filename):
	result = []
	for i in range(max_nb_chars+1): result.append({})

	with open(filename, 'r', encoding='UTF-8') as f:
		text = f.read()
		for line in text.splitlines()[1:]:
			fields = line.split(' ')
			for word in fields:
				word = word.lower().replace('œ' ,'oe').replace('\x9c' ,'oe')
				word_normalized = ''.join(c for c in unicodedata.normalize('NFD', word) if unicodedata.category(c) != 'Mn')
				word_normalized = word_normalized.replace('-' ,'').replace('\'','').replace('¹', '').replace('²', '').replace('³', '')
				n = len(word_normalized)
				if n > max_nb_chars or not word_normalized.isalpha() or not word_normalized.isascii():
					continue
				
				# Ajout le mot dans la base, avec une probabilité de 0.01%
				if word_normalized not in result[n]:
					result[n][word_normalized] = default_freq

	return result

def import_csv_worldlex(filename, existing_db, freq_field=None):
	new_db = []
	for i in range(max_nb_chars+1): new_db.append({})
	with open(filename, 'r', encoding='UTF-8') as f:
		text = f.read()
		for line in text.splitlines()[1:]:
			fields = line.split('\t')
			if freq_field:
				freq = float(fields[freq_field])
			else:
				freq = sum([float(fields[x]) for x in [2, 6, 10]]) / 3.0 # 2=blog, 6=twitter, 10=news

			word = fields[0].lower().replace('œ' ,'oe').replace('\x9c' ,'oe')
			word_normalized = ''.join(c for c in unicodedata.normalize('NFD', word) if unicodedata.category(c) != 'Mn')
			word_normalized = word_normalized.replace('-' ,'').replace('\'','').replace('¹', '').replace('²', '').replace('³', '')
			n = len(word_normalized)
			if n > max_nb_chars or not word_normalized.isalpha() or not word_normalized.isascii():
				# print('rejected', word_normalized)
				continue
			
			# Ajout le mot dans la base, en ajoutant la fréquence à d'autres formes éventuelles
			new_db[n][word_normalized] = freq + new_db[n].get(word_normalized, 0)
	
	# Merge
	result = []
	for n in range(max_nb_chars+1):
		merged = existing_db[n].copy() # avec "default_freq"
		for word, proba in new_db[n].items():
			# Penalise les mots qui ne sont pas dans la liste scrabble, favorise ceux qui y sont
			merged[word] = max(proba, default_freq) if word in existing_db[n] else min(proba, default_freq/100)
		result.append(merged)

	return result

def save_to_json(result, json_name):
	for i in range(1, max_nb_chars+1):
		most_probable_word = max(result[i].keys(), key = lambda x: result[i][x]) if result[i] else ''
		print(f'Mots à {i} lettres: {len(result[i])} ({most_probable_word})')

	# Write whole dict
	with open(f'{json_name}.json', 'w') as f:
		json.dump(result, f)

	# Write small dict (1 per nb of letters)
	for i in range(3, max_nb_chars+1):
		if len(result[i]) < 10:
			continue
		with open(f'{json_name}_{i}.json', 'w') as f:
			json.dump({'words': result[i]}, f)


###############################################################################

# O = rien, 1 = bonne lettre mauvaise place, 2 = bonne lettre, bonne place
def compute_result(proposed_word, solution_word):
	result = [0] * len(proposed_word)
	prop_l, sol_l = list(proposed_word), list(solution_word)
	# Cherche les lettres bien placées
	for i, c in enumerate(prop_l):
		if c == sol_l[i]:
			result[i] = 2
			prop_l[i] = sol_l[i] = None

	# Parmi les lettres restantes, cherche les lettres mal placées
	for i, c in enumerate(prop_l):
		if c is not None and c in sol_l:
			sol_l.remove(c)
			result[i] = 1

	return result

def convert_result(result_list):
	result_int = [r * b for r, b in zip(result_list, base)]
	return sum(result_int)

def compute_entropy(dico_n):
	dico_values = np.array(list(dico_n.values()))
	dico_values = dico_values[dico_values!=0] / dico_values.sum()
	entropy = -(dico_values * np.log2(dico_values)).sum()
	return entropy

def distribution_possibilities(proposed_word, dico_n):
	distribution = np.zeros(3**(max_nb_chars+1))
	for solution_word, prob in dico_n.items():
		result = convert_result( compute_result(proposed_word, solution_word) )
		distribution[result] += prob
	# compute entropy
	distribution_norm = distribution[distribution!=0] / distribution.sum()
	entropy = -(distribution_norm * np.log2(distribution_norm)).sum()
	return distribution, entropy

def expected_remaining_moves(dico_solutions, dico_admissible):
	total = sum([p for _,p in dico_solutions.items() if p > 0])
	current_entropy = compute_entropy(dico_solutions)
	print(f'   entropie = {current_entropy:.02f} bit')
	if len(dico_solutions) * len(dico_admissible) > max_combinations_to_try:
		nb_words = max(min_words_to_try, max_combinations_to_try//len(dico_solutions))
		sorted_prob = sorted(dico_admissible.values(), reverse=True)
		dico_admissible = {w:p for w,p in dico_admissible.items() if p >= sorted_prob[nb_words]}
		print(f'   trop de mots à essayer, je teste les {len(dico_admissible)} plus populaires')	
	def estimator(w, p):
		p_normalized = p/total
		_, entropy = distribution_possibilities(w, dico_solutions)
		remaining_moves = round( (1-p_normalized) * (1.24+(current_entropy-entropy)*0.71) , 4)
		#return remaining_moves, current_entropy-entropy
		return remaining_moves

	entropy_list = [ (word, estimator(word, prob if word in dico_solutions else 0)) for word, prob in tqdm(dico_admissible.items(), ncols=60, leave=False, disable=(sys.platform == 'emscripten')) ]
	entropy_list = sorted(entropy_list, key=lambda x: x[1])
	return entropy_list

def remove_based_on_result(dico_n, proposed_word, result_int):
	new_dico_n = {
		w: p for w, p in dico_n.items()
			if convert_result(compute_result(proposed_word, w)) == result_int
	}
	print(f'   taille du dictionnaire {len(dico_n)} --> {len(new_dico_n)}. Mots les plus populaires: {sorted(new_dico_n.keys(), key=lambda x: new_dico_n[x], reverse=True)[:5]}')
	return new_dico_n

def remove_based_on_first_letter(dico_n, c):
	new_dico_n = {
		w: p for w, p in dico_n.items() if w[0] == c
	}
	print(f'   taille du dictionnaire {len(dico_n)} -> {len(new_dico_n)} {sorted(new_dico_n.keys(), key=lambda x: new_dico_n[x], reverse=True)[:5]}')
	return new_dico_n

def simulation(word_to_find, dico_n):
	dico_solutions = dico_n
	best_word_to_try = None
	while best_word_to_try != word_to_find:
		best_moves = expected_remaining_moves(dico_solutions, dico_n)
		best_word_to_try, nb_moves = best_moves[0]
		if len(best_moves) > 3:
			print(f'je tente "{best_word_to_try.upper()}" avec {nb_moves} coup(s) estimés, il y avait aussi {best_moves[1]} et {best_moves[2]}')
		else:
			print(f'je tente "{best_word_to_try.upper()}" avec {nb_moves} coup(s) estimés')
		result = convert_result(compute_result(best_word_to_try, word_to_find))
		dico_solutions = remove_based_on_result(dico_solutions, best_word_to_try, result)

def online_simulation(dico_n):
	user_input = input('Une idee de la premiere lettre, ou proposition de mot déjà faite (vide sinon) ? ')
	if len(user_input) == 1:
		dico_n = remove_based_on_first_letter(dico_n, user_input)
		dico_solutions = dico_n
	elif len(user_input) > 1:
		word_trial, trial_result = parse_user_input(user_input)
		dico_solutions = remove_based_on_result(dico_n, word_trial, trial_result)
	else:
		dico_solutions = dico_n

	while len(dico_solutions) > 1:
		best_moves = expected_remaining_moves(dico_solutions, dico_n)
		best_word_to_try, nb_moves = best_moves[0]
		if len(best_moves) > 3:
			print(f'je tente "{best_word_to_try.upper()}" avec {nb_moves} coup(s) estimés, il y avait aussi {best_moves[1]} et {best_moves[2]}')
		else:
			print(f'je tente "{best_word_to_try.upper()}" avec {nb_moves} coup(s) estimés')
		best_word_to_try, result = parse_user_input(best_word_to_try)
		dico_solutions = remove_based_on_result(dico_solutions, best_word_to_try, result)

# Used by browser when importing this module
def online_simulation_browser_filter(dico_n):
	import js
	dico_solutions = dico_n
	trials  = [js.document.getElementById("try_"+str(i)).value for i in range(5)]
	results = js.results
	print(trials, results)

	# Digest user inputs
	for i, (word_trial, trial_result) in enumerate(zip(trials, results)):
		if len(word_trial) == int(js.n_total_char):
			print('Input used:', word_trial, trial_result)
			dico_solutions = remove_based_on_result(dico_solutions, word_trial, trial_result)
		else:
			if len(word_trial) > 0 or trial_result > 0:
				print('Input not used:', word_trial, trial_result)
				# message += f'Warning: input not used -> {word_trial} {trial_result}'
	# message += f'{len(dico_solutions)} mot(s) possible(s) dont {sorted(dico_solutions.keys(), key=lambda x: dico_solutions[x], reverse=True)[:5]}'
	best_words = ', '.join(sorted(dico_solutions.keys(), key=lambda x: dico_solutions[x], reverse=True)[:5])
	js.hint_output.innerHTML = f'{len(dico_solutions)} mot(s) possible(s) dont: {best_words}'

	return dico_solutions

# Used by browser when importing this module
def online_simulation_browser_best(dico_solutions, dico_n):
	import js
	message = ''

	# Now compute 
	best_moves = expected_remaining_moves(dico_solutions, dico_n)
	best_word_to_try, nb_moves = best_moves[0]

	# Display
	if len(best_moves) > 3:
		message += f' - je tente "{best_word_to_try.upper()}" avec {nb_moves} coup(s) estimés, il y avait aussi {best_moves[1]} et {best_moves[2]}'
	else:
		message += f' - je tente "{best_word_to_try.upper()}" avec {nb_moves} coup(s) estimés'

	i_last_word = max([i for i in range(5) if len(js.document.getElementById("try_"+str(i)).value) == int(js.n_total_char)], default=-1)
	js.document.getElementById("try_"+str(i_last_word+1)).value = best_word_to_try.lower()
	js.updateButtonsContent(i_last_word+1)
	return message

def parse_user_input(word_trial):
	while True:
		input_string = input('Résultat ? (1=mal placé, 2=bien placé, 0=sinon, x=arrêter ou bien autre mot). Ex: "00211":   ')
		input_string = input_string.rstrip().lstrip()
		if input_string == 'x':
			exit(1)

		try:
			if input_string.isalpha():
				if len(input_string) == len(word_trial):
					word_trial = input_string
					print(f'  on change le mot essayé pour {word_trial}')
					continue # Now enter result
			else:
					list_digits = [int(x) for x in list(input_string)]
					if len(list_digits) != len(word_trial) or min(list_digits) < 0 or max(list_digits) > 2:
						raise Exception('bad digits')
		except:
			continue

		return word_trial, convert_result(list_digits)

def load_dico_local(json_name, nb_letters):
	dico = json.load(open(join(dirname(__file__), f'{json_name}_{nb_letters}.json')))
	return dico['words'], dico.get('precomputed')

# Used by browser when importing this module
async def load_dico_remote(lang, nb_letters):
	from pyodide.http import pyfetch
	json_name = f'https://raw.githubusercontent.com/cestpasphoto/wordle_solver_french/main/dict_{lang}_{nb_letters}.json'
	response = await pyfetch(json_name)
	dico = await (response.json())
	return dico['words']

def adjust_dico(dico, top_words_only, prob):
	# Adjust words probability
	if top_words_only:
		dico = {w: p for w,p in dico.items() if p >= default_freq}
	if prob == 'sqrt' or prob == 'average':
		dico = {w: sqrt(p) for w,p in dico.items()}   						# Applatit histogramme de fréq des mots
	elif prob == 'equal' or prob == 'hard':
		dico = {w: 1+p/100 for w,p in dico.items()}   						# (quasi) équiprobabilité, tout en gardant la possibilité d'ordonner
	elif prob == 'sqrt_nosmall':
		dico = {w: sqrt(max(p,default_freq/100)) for w,p in dico.items()}   # Applatit histogramme de fréq des mots
	elif prob == 'nosmall':
		dico = {w: max(p,default_freq/100) for w,p in dico.items()}   		# Booste la probabilité des mots les moins courants

	return dico

###################################################################################

def main():
	import argparse
	parser = argparse.ArgumentParser(description='wordle guesser')
	parser.add_argument('--fr'         , action='store_true', help='Use french dictionnary (default)')
	parser.add_argument('--en'         , action='store_true', help='Use english dictionnary')
	parser.add_argument('--build-dict' , action='store_true', help='Build dictionnary from list of words')
	parser.add_argument('--words', '-w', default='top', choices=['all', 'top'], help='Which words to use: "all" or "top" (default)')
	parser.add_argument('--prob',  '-p', default='hard', choices=['average', 'hard', 'original', 'sqrt', 'equal', 'nosmall', 'sqrt_nosmall'], help='Choose words\
probabilities: either original ones or bump very small probabilities or flattened a bit\
(sqrt - good for AVERAGE difficulty games) or make it completely flat (equal - good for HARD games).')
	parser.add_argument('word_to_guess', nargs='?', default='', help='If known, provide the word to guess to run non-interactive game')
	args = parser.parse_args()

	json_name = 'dict_en' if args.en and not args.fr else 'dict_fr'
	if args.build_dict:
		if args.en:
			dico = import_csv_scrabble('Collins Scrabble Words (2019).txt')	# https://drive.google.com/file/d/1oGDf1wjWp5RF_X9C7HoedhIWMh5uJs8s/view
			dico = import_csv_worldlex('Eng_US.Freq.2.txt', dico)			# http://www.lexique.org/?page_id=250
		else:
			dico = import_csv_scrabble('touslesmots.txt') 		# https://www.listesdemots.net/touslesmots.txt
			dico = import_csv_worldlex('Fre.Freq.2.txt', dico)  # http://www.lexique.org/?page_id=250
		save_to_json(dico, json_name)
		print('les dictionnaires sont maintenant prêts')
		return


	# Load dictionnary
	dico, _ = load_dico_local(json_name, len(args.word_to_guess) if args.word_to_guess else int(input('Combien de lettres ?  ')))
	dico = adjust_dico(dico, args.words == 'top', args.prob)

	if args.word_to_guess:
		if input('On suppose connaitre la 1e lettre ? (o/n):  ').lower() == 'o':
			dico = remove_based_on_first_letter(dico, args.word_to_guess[0])
		simulation(args.word_to_guess, dico)
	else:
		online_simulation(dico)

if __name__ == "__main__":
	main()