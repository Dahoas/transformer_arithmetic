from copy import copy
import re


"""
Things to remove in processing
1. Functions with bad visibility
2. copy of TInt 
3. base function names
"""

class TInt:
  # Dict tracking list of functions to make visible in trace
  visibility = {}

  def __init__(self, val, vis=False):
    """
    Remove leading 0s and store in val
    Note "" is 0 is 00000
    """
    self.val = str(val)

  def __repr__(self):
    return "Val: {}, Int: {}".format(self.val, int(self))

  def len(x, vis=False):
    """
    Implements len for TInt to return a TInt. Must be defined this way as 
    __len__ expects an int to be returned

    Note 0 has length 0.
    """
    return TInt(len(x.val))

  def __int__(self, vis=False):
    """
    Converts TInt to int. Adds 0 in front 
    to deal with equality of 0 and ""
    """
    return int(self.val) if self.val != "" else 0

  def __eq__(x, y, vis=False):
    return int(x) == int(y)

  def __ne__(x, y, vis=False):
    return int(x) != int(y)

  def __gt__(x, y, vis=False):
    return int(x) > int(y)

  def __ge__(x, y, vis=False):
    return int(x) >= int(y)

  def __or__(x, y, vis=False):
    """
    Implements TInt concatenation
    """
    return TInt(x.val + y.val)

  def __is_zero(self):
    return self.val == ""

  def __getitem__(self, ind, vis=False):
    if ind >= self.len():
      return copy(O)
    else:
      return TInt(self.val[len(self.val) - int(ind) - 1])

  def drop(x, vis=False):
    """
    Drops the units digit from x. Dropping 0 gives 0.
    """
    if x.len() == I:
      return copy(O)
    return TInt(x.val[:-1])

  def __add(x, y, vis=False):
    """
    Implements basic addition. __ denotes hidden from user.
    """
    assert x.len() <= I
    assert y.len() <= I
    return TInt(int(x) + int(y))
  
  def __add__(x, y, vis=True):
    if x.len() <= I and y.len() <= I:
      return x.__add(y)
    else:
      res = copy(O)
      carry = copy(O)
      while x != O or y != O:
        num1 = x[O]
        num2 = y[O]
        x = x.drop()
        y = y.drop()
        ds = num1 + num2
        ds = ds + carry
        res = ds[O] | res
        carry = copy(O) if ds.len() <= I else copy(I)
      res = carry | res
      return res

  def __lshift__(x, y, vis=False):
    """
    Implements left shifting for TInt
    Note 0 -> 00
    """
    return TInt(x.val + "0")

  def __mul(x, y, vis=False):
    """
    Implements basic multiplication. __ denotes hidden from user.

    Note: This might be a difficult base case to learn
    """
    assert x.len() <= I
    assert y.len() <= I
    return TInt(int(x) * int(y))

  def __mul__(x, y, vis=True):
    # If x shorter than y swap x and y. x is now always larger
    if x.len() < y.len():
      t = x
      x = y
      y = t
    # Base case: both numbers are single digit
    if x.len() <= I and y.len() <= I:
      return x.__mul(y)
    else:
      out_res = copy(O)
      carry = copy(O)
      mag = copy(O)
      # Outer loop for multiplying with each digit of y
      while y != O:
        fac = y[O]
        y = y.drop()
        x_c = copy(x)
        in_res = copy(O)
        # Inner loop for multiplying
        while x_c != O:
          term = x_c[O]
          x_c = x_c.drop()
          dm = fac * term
          dm = dm + carry
          in_res = dm[O] | in_res
          carry = copy(O) if dm.len() <= I else dm[I]
        # Add any residual carry
        in_res = carry | in_res
        carry = copy(O)
        # Shift in_res by mag and add to out_res
        in_res = in_res | mag
        mag = mag << I
        out_res = out_res + in_res
      return out_res

  def __floordiv__(x, y, vis=False):
    """
      Implements x // y using grade-school long division
    """
    quo = copy(O)
    div_len = y.len()
    while x >= y:
      mag_diff = x.len() - div_len
      
    return quo

O = TInt("")
I = TInt("1")

x = TInt(45345)
y = TInt(925634)
z = x * y
print(z)
