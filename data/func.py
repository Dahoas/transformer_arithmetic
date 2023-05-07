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
  #val = "0"

  def __init__(self, val, vis=False):
    """
    Remove leading 0s and store in val
    Note "" is 0 is 00000
    """
    self.val = str(val)

  def __repr__(self, vis=False):
    return self.val

  def __str__(self, vis=False):
    return self.val

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
      res = x.__add(y)
    else:
      res = copy(O)
      carry = copy(O)
      while x != O or y != O:
        digx = x[O]
        digy = y[O]
        x = x.drop()
        y = y.drop()
        ds = digx + digy
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
    return TInt(x.val + ("0" * int(y)))

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
      out_res = x.__mul(y)
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

  def __sub(x, y, vis=False):
    assert x <= TInt(18)
    assert y <= TInt(18)
    return TInt(int(x) - int(y))

  def __sub__(x, y, vis=True):
    #print("SUBTRACT: ", x, y)
    assert x >= y
    if x <= TInt(18) and y <= TInt(18):
      #print("basecase")
      res = x.__sub(y)
    #print("not basecase")
    else:
      res = copy(O)
      borrow = copy(O)
      while y != O:
        #print("looping...", x, y, borrow, res)
        # get next digis
        digx = x[O]
        digy = y[O]
        x = x.drop()
        y = y.drop()
        # factor in borrow
        #print("got digx, digy, borrow: ", digx, digy, borrow)
        if borrow > O:
          if digx > O:
            digx = digx - I
            borrow = O
          else:
            digx = I | TInt(0)
            digx = digx - I
        #print("facored in borrow: digx, borrow", digx, borrow)
        # subtract
        if digx < digy:
          assert borrow == O
          digx = I | digx
          borrow = I
        #print("borrowed as needed digx, borrow", digx, borrow)
        dd = digx - digy
        res = dd | res
        #print("loop done. x, y, borrow, res:", x, y, borrow, res)
      while borrow != O:
        digx = x[O]
        x = x.drop()
        if digx > O:
          digx = digx - I
          borrow = O
        else:
          digx = I | TInt(0)
          digx = digx - I
        res = digx | res
      res = x | res
    #print("got res:", res)
    return res

  def __floordiv__(x, y, vis=True):
    """
      Implements x // y using grade-school long division
    """
    res = copy(O)
    while x >= y:
      # choose factor
      #print("div x, y, res: ", x, y, res)
      """factor_mag = x.len() - y.len() - I # TODO - this require
      if (y << (factor_mag)) > x:
        factor_mag = factor_mag - I
      #print(y << factor_mag, x)"""
      factor_mag = I
      while (y << factor_mag) <= x:
        factor_mag = factor_mag + I
        #print(y << factor_mag)
      factor_mag = factor_mag - I
      #print(x, y, factor_mag)
      dig = I
      sum_div = y
      #print("found_mag, dig, sum_div: ", factor_mag, dig, sum_div)
      while (sum_div + y) << factor_mag <= x:
        #print("finding dig, sum_div: ", dig, sum_div)
        #print(sum_div | factor_mag)
        dig = dig + I
        sum_div = sum_div + y
      #print("factor sum_div, dig: ", sum_div << factor_mag, dig)
      factor = dig << factor_mag
      # add factor to res
      res = res + factor
      x = x - (sum_div << factor_mag)
      #print("factor, res, x:", factor, res, x)
    return res

Of = TInt("0")
O = TInt("")
I = TInt("1")

#x = TInt(405600)
#y = TInt(39)
#print(x - y)

x = TInt(409500040)
y = TInt(17)
z = x // y
print(z)
