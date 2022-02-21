# Wordle solver in French
A wordle solver with French dictionnary using Python based on
- [3b1b video](https://www.youtube.com/watch?v=v68zYyaEmEA), using entropy maximisaton
- French dictionnaries https://www.listesdemots.net/touslesmots.txt (Scrabble) and http://www.lexique.org/?page_id=250 (frequency of words)

# How to use it?

Simply run `python3 wordle.py`: it will ask you how many letters and whether first letter is known. Then you need to give result. If you already know the word, you can use `python3 wordle.py your_word`.

First run should use side pickle which contains dictionnary, otherwise you need to download both links above and it will build such pickle and store it.
During search, it will max out number of tested combinations