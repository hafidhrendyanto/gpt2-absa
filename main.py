from transformers import TFAutoModelWithLMHead
from gpt2absa.constant import restaurant_aspect_categories, laptop_aspect_categories
from gpt2absa import aspect_polarity_pair

def inputdomain():
  inp = None
  while inp is None:
    inp = input()
    try:
      inp = int(inp)
    except ValueError:
      if inp.lower() == 'restaurant':
        inp = 1
      elif inp.lower() == 'laptop':
        inp = 2
      else:
        print("Your choice must be either laptop or restaurant")
        inp = None

  return inp

def main():
  model = TFAutoModelWithLMHead.from_pretrained("hafidhrendyanto/gpt2-absa")
  print('''
GPT-2 ASPECT BASED SENTIMENT ANALYSIS
This is a demo application of an aspect based sentiment analyisis solution that uses GPT-2 as it's language processing model.
You will enter a review text, and the system will output it's predicted aspect and sentiment pair.
The GPT-2 model is fine tuned using SemEval 2016 Task 5 Restaurant & Laptop dataset, as such this aplication 
can only be tested on either restaurant or laptop dataset. 
Please pick the domain of your data from the option below:
1. Restaurant
2. Laptop
  ''')
  inp = inputdomain()

  if inp == 1:
    aspect_categories = restaurant_aspect_categories
  else:
    aspect_categories = laptop_aspect_categories

  print('''
Please enter the review text that you want to test below
You exit by typing QUIT
  ''')

  finish = False
  while not finish:
    text = input()
    if text != "QUIT":
      ress = aspect_polarity_pair(model, text, aspect_categories)
      print(ress)
    else:
      finish = True

if __name__ == '__main__':
    main()