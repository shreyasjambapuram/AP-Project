import sys
import datetime
from typing import List

import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt



def ask():
	c = ""
	lst = []
	while(c!="stop"):
		c = str(input("Input: " ))
		if(len(c)==4 and c !="stop"):
			c = c.upper()
			lst.append(c)
		else:
			print("Type A valid Input")
	return lst

lst = ask()
print(lst)
