#!/usr/bin/python3

import unicodedata
import pickle
from math import log2, sqrt
import numpy as np
from tqdm import tqdm
import sys

max_nb_chars = 10
base = [3**i for i in range(max_nb_chars+1)]
max_combinations_to_try = 2000000
default_freq = 0.001

estimates = []
archives = []

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

def save_to_pickle(result):
	for i in range(1, max_nb_chars+1):
		most_probable_word = max(result[i].keys(), key = lambda x: result[i][x]) if result[i] else ''
		print(f'Mots à {i} lettres: {len(result[i])} ({most_probable_word})')
	with open(pickle_name, 'wb') as f:
		pickle.dump(result, f, pickle.HIGHEST_PROTOCOL)


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
	distribution = distribution[distribution!=0] / distribution.sum()
	entropy = -(distribution * np.log2(distribution)).sum()
	return distribution, entropy

def expected_remaining_moves(dico_solutions, dico_admissible):
	total = sum([p for _,p in dico_solutions.items() if p > 0])
	current_entropy = compute_entropy(dico_solutions)
	print(f'   entropie = {current_entropy:.02f} bit')
	if len(dico_solutions) * len(dico_admissible) > max_combinations_to_try:
		nb_words = max(200, max_combinations_to_try//len(dico_solutions))
		sorted_prob = sorted(dico_admissible.values(), reverse=True)
		dico_admissible = {w:p for w,p in dico_admissible.items() if p > sorted_prob[nb_words]}
		print(f'   trop de mots à essayer, je teste les {len(dico_admissible)} plus populaires')	
	def estimator(w, p):
		p_normalized = p/total
		_, entropy = distribution_possibilities(w, dico_solutions)
		remaining_moves = round( (1-p_normalized) * (1.24+(current_entropy-entropy)*0.71) , 2)
		#return remaining_moves, current_entropy-entropy
		return remaining_moves

	entropy_list = [ (word, estimator(word, prob if word in dico_solutions else 0)) for word, prob in tqdm(dico_admissible.items(), ncols=60, leave=False) ]
	entropy_list = sorted(entropy_list, key=lambda x: x[1])
	return entropy_list

def remove_based_on_result(dico_n, proposed_word, result_int):
	new_dico_n = {
		w: p for w, p in dico_n.items()
			if convert_result(compute_result(proposed_word, w)) == result_int
	}
	print(f'   taille du dictionnaire {len(dico_n)} -> {len(new_dico_n)} : {sorted(new_dico_n.keys(), key=lambda x: new_dico_n[x], reverse=True)[:5]}')
	return new_dico_n

def remove_based_on_first_letter(dico_n, c):
	new_dico_n = {
		w: p for w, p in dico_n.items() if w[0] == c
	}
	print(f'   taille du dictionnaire {len(dico_n)} -> {sorted(new_dico_n.keys(), key=lambda x: new_dico_n[x], reverse=True)[:5]}')
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
		#estimates.append(entropy)
		result = convert_result(compute_result(best_word_to_try, word_to_find))
		dico_solutions = remove_based_on_result(dico_solutions, best_word_to_try, result)

	# Archivage infos
	#for i, entropy in enumerate(estimates[::-1]):
	#	archives.append( (entropy, i+1) )

def online_simulation(dico_n):
	user_input = input('Une idee de la premiere lettre, ou proposition de mot déjà faite (vide sinon) ? ')
	if len(user_input) == 1:
		dico_n = remove_based_on_first_letter(dico_n, user_input)
		dico_solutions = dico_n
	elif len(user_input) > 1:
		result = convert_result(parse_user_input(len(user_input)))
		dico_solutions = remove_based_on_result(dico_n, user_input, result)
	else:
		dico_solutions = dico_n

	while len(dico_solutions) > 1:
		best_moves = expected_remaining_moves(dico_solutions, dico_n)
		best_word_to_try, nb_moves = best_moves[0]
		if len(best_moves) > 3:
			print(f'je tente "{best_word_to_try.upper()}" avec {nb_moves} coup(s) estimés, il y avait aussi {best_moves[1]} et {best_moves[2]}')
		else:
			print(f'je tente "{best_word_to_try.upper()}" avec {nb_moves} coup(s) estimés')
		#estimates.append(entropy)
		result = convert_result(parse_user_input(len(best_word_to_try)))
		dico_solutions = remove_based_on_result(dico_solutions, best_word_to_try, result)

	# Archivage infos
	#for i, entropy in enumerate(estimates[::-1]):
	#	archives.append( (entropy, i+1) )


def parse_user_input(length):
	while True:
		input_string = input('Résultat ? (1=mal placé, 2=bien placé, 0=sinon, x pour arrêter). Ex: "00211":   ')
		if input_string == 'x':
			exit(1)
		list_digits = list(input_string.rstrip().lstrip())
		try:
			list_digits = [int(x) for x in list_digits]
			if len(list_digits) != length or min(list_digits) < 0 or max(list_digits) > 2:
				raise Exception('bad digits')
		except:
			continue

		return list_digits

###################################################################################

pickle_name = 'parsed_fr_dictionary.pickle'
try:
	dico = pickle.load(open(pickle_name, 'rb'))
except (OSError, IOError) as e:
	dico = import_csv_scrabble('touslesmots.txt') 		# https://www.listesdemots.net/touslesmots.txt
	dico = import_csv_worldlex('Fre.Freq.2.txt', dico)  # http://www.lexique.org/?page_id=250
	save_to_pickle(dico)
	print('les dictionnaires sont maintenant charges')

# import random
# for _ in range(20):
# 	for _ in range(10):
# 		n = random.randrange(3,max_nb_chars+1)
# 		dico_n = dico[n]
# 		solution = random.choices(list(dico_n.keys()), weights=[int(p>10*default_freq) for _,p in dico_n.items()])[0]
# 		print(solution)
# 		dico_n = {w: max(p,default_freq/100) for w,p in dico_n.items()}
# 		simulation(solution, dico_n)
# 		estimates = []
# 	for e in range(30):
# 		e_min, e_max = (e-5)/10, (e+5)/10
# 		nb_coups = [ i for e_,i in archives if e_min<=e_<e_max ]
# 		nb_coups_avg = sum(nb_coups)/len(nb_coups) if nb_coups else 0.
# 		print(f'{e/10};{nb_coups_avg}')

if len(sys.argv) > 1:
	solution = sys.argv[1]
	dico = dico[len(solution)]
	# dico = {w: sqrt(max(p,default_freq/100)) for w,p in dico.items()}   # Applatit histogramme de fréq des mots
	dico = {w: max(p,default_freq/100) for w,p in dico.items()}
	if input('On suppose connaitre la 1e lettre ? (o/n):  ').lower() == 'o':
		dico = remove_based_on_first_letter(dico, solution[0])
	simulation(solution, dico)
	# print(archives)
else:	
	dico = dico[int(input('Combien de lettres ?  '))]
	# dico = {w: sqrt(max(p,default_freq/100)) for w,p in dico.items()} # Applatit histogramme de fréq des mots
	dico = {w: max(p,default_freq/100) for w,p in dico.items()}

	online_simulation(dico)
