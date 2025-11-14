import sys
import datetime
from typing import List

# Third-party dependencies
try:
	import yfinance as yf
	import pandas as pd
	import matplotlib.pyplot as plt
except ModuleNotFoundError as e:
	print(f"Missing package: {e.name}. Install with: python -m pip install yfinance pandas matplotlib")
	sys.exit(1)

# Required third-party packages: yfinance, pandas, matplotlib


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
