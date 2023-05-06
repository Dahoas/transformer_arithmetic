def sort_integers(l):
  result = []
  while len(l) > 0:
    greatest_ind = 0
    greatest_val = l[0]
    for i in range(len(l)):
      if l[i] > greatest_val:
        greatest_ind = i
        greatest_val = l[i]
    result = [greatest_val] + result
    l = l[:greatest_ind] + l[greatest_ind + 1:]
  return result

def add(x : str, y : str):
  res = ""
  carry = 0
  max_len = max(len(x), len(y))
  x = x.rjust(max_len, "0")
  y = y.rjust(max_len, "0")
  for i in range(max_len):
    num1 = x[max_len - i - 1]
    num2 = y[max_len - i - 1]
    ds = str(int(num1) + int(num2))
    ds = str(int(ds) + carry)
    if len(ds) > 1:
      carry = 1
      ds = ds[1]
    else:
      carry = 0
    res = ds + res
  if carry > 0:
    res = "1" + res
  return res

def is_digit(x):
  return x in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

def length(x):
  if x == '': return "0"
  if x in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]:
    return "1"
  else:
    return str(int(length(x[1:])) + 1)

def greater_than(x, y):
  if x == '': return "0"
  if y == '': return "1"
  if (is_digit(x) and not is_digit(y)): return "0"
  if (is_digit(y) and not is_digit(x)): return "1"
  if (is_digit(x) and is_digit(y)): 
    if int(x) > int(y): return "1"
    else: return "0"
  x_len = length(x)
  y_len = length(y)
  if greater_than(x_len, y_len): return "1"
  elif greater_than(y_len, x_len): return "0"

  while not x == '':
    dig1 = x[0]
    dig2 = y[0]
    if dig1 > dig2: return "1"
    if dig2 > dig1: return "0"
    x = x[1:]
    y = y[1:]
  return "0"

def subtract(x : str, y : str):
  #assert(greater_than(y, x) == "0")
  res = ""
  borrowed = "0"

  while greater_than(length(y), "0") == "1":
    dig1 = x[-1]
    dig2 = y[-1]
    if greater_than(borrowed, "0") == "1":
      if greater_than(dig1, "0") == "1":
        dig1 = str(int(dig1) - int(borrowed))
        borrowed = "0"
      else:
        dig1 = "9"
        borrowed = "1"
    if greater_than(dig2, dig1) == "1":
      borrowed = "1"
      dig1 = "1" + dig1
    dig_res = str(int(dig1) - int(dig2))
    res = dig_res + res
    x = x[:-1]
    y = y[:-1]

  while greater_than(borrowed, "0") == "1":
    dig1 = x[-1]
    if greater_than(dig1, "0") == "1":
      dig1 = str(int(dig1) - int(borrowed))
      borrowed = "0"
    else:
      dig1 = "9"
    x = x[:-1]
    if not x == '' or not dig1 == "0": res = dig1 + res

  if greater_than(x, "0") == "1":
    res = x + res

  return res
    
print(subtract("1000", "1"))
print(subtract("9000", "1"))
print(subtract("333", "222"))
print(subtract("303", "4"))
