from copy import copy
import re
from enum import Enum
import functools


"""
Things to remove in processing
1. Functions with bad visibility
2. copy of TInt
3. base function names
"""

# Invis functions generate no trace
INVIS = 0
# Visible functions generate a trace
VIS = 1
# Called functions are wrapped in call(...) with no trace
CALL = 2


class TInt:
  # Dict tracking list of functions to make visible in trace
  val = "0"
  VIS_DICT = {
              "__repr__": INVIS,
              "__str__": INVIS,
              "len": INVIS,
              "__int__": INVIS,
              "__eq__": INVIS,
              "__ne__": INVIS,
              "__gt__": INVIS,
              "__ge__": INVIS,
              "__or__": INVIS,
              #"__is_zero": INVIS,
              "__getitem__": INVIS,
              #"__add": INVIS,
              "__add__": VIS,
              "__rshift__": INVIS,
              "__lshift__": INVIS,
              #"__mul": INVIS,
              "__mul__": VIS,
              #"__sub": INVIS,
              "__sub__": VIS,
              "__floordiv__": VIS,
            }

  def __init__(self, val, vis=INVIS):
    """
    Remove leading 0s and store in val
    Note "" is 0 is 00000
    """
    self.val = str(val)

  @classmethod
  def update_vis(cls, f_name, vis):
    """Updates default visibility of input function f_name
    """
    setattr(cls, f_name, functools.partialmethod(getattr(TInt, f_name), vis=vis))

  @classmethod
  def reset_vis(cls):
    for f_name in cls.VIS_DICT:
      cls.update_vis(f_name, cls.VIS_DICT[f_name])

  def __repr__(self, vis=INVIS):
    if self.val == '': return "0"
    return self.val

  def __str__(self, vis=INVIS):
    if self.val == '': return "0"
    return self.val

  def len(x, vis=INVIS):
    """
    Implements len for TInt to return a TInt. Must be defined this way as
    __len__ expects an int to be returned

    Note 0 has length 0.
    """
    if x.val == '': return 0
    return TInt(len(str(int(x.val))))

  def __int__(self, vis=INVIS):
    """
    Converts TInt to int. Adds 0 in front
    to deal with equality of 0 and ""
    """
    return int(self.val) if self.val != "" else 0

  def __eq__(x, y, vis=INVIS):
    return int(x) == int(y)

  def __ne__(x, y, vis=INVIS):
    return int(x) != int(y)

  def __gt__(x, y, vis=INVIS):
    return int(x) > int(y)

  def __ge__(x, y, vis=INVIS):
    return int(x) >= int(y)

  def __or__(x, y, vis=INVIS):
    """
    Implements TInt concatenation
    """
    return TInt(x.val + y.val)

  def __is_zero(self, vis=INVIS):
    return self.val == ""

  def __getitem__(self, ind, vis=INVIS):
    if ind >= self.len():
      return copy(O)
    else:
      return TInt(self.val[len(self.val) - int(ind) - 1])

  def __add(x, y, vis=INVIS):
    """
    Implements basic addition. __ denotes hidden from user.
    """
    return TInt(int(x) + int(y))

  def __add__(x, y, vis=VIS):
    res = copy(E)
    carry = copy(O)
    while x != O or y != O:
      #print(x, y, res)
      digx = x[O]
      digy = y[O]
      x = x >> I
      y = y >> I
      ds = digx.__add(digy)
      ds = ds.__add(carry)
      res = ds[O] | res
      if ds.len() <= I:
        carry = copy(O)
      else:
        carry = copy(I)
      #print(x, y, res)
    if carry > O:
      res = carry | res
    return res

  def __rshift__(x, y, vis=INVIS):
    """
    Drops the units digit from x. Dropping 0 gives 0.
    """
    #print("rsh", x, y)
    if x.len() <= y:
      #print("rsh xlen <= y -> return 0")
      return copy(O)
    elif y == O:
      return copy(x)
    #print("rsh ", str(int(x.val))[:-1 * int(y)])
    return TInt(str(int(x.val))[:-1 * int(y)])


  def __lshift__(x, y, vis=INVIS):
    """
    Implements left shifting for TInt
    Note 0 -> 00
    """
    return TInt(x.val + ("0" * int(y)))

  def __mul(x, y, vis=INVIS):
    """
    Implements basic multiplication. __ denotes hidden from user.

    Note: This might be a difficult base case to learn
    """
    assert x.len() <= I
    assert y.len() <= I
    return TInt(int(x) * int(y))

  def __mul__(x, y, vis=VIS):
    # If x shorter than y swap x and y. x is now always larger
    if x.len() < y.len():
      t = x
      x = y
      y = t
    # Base case: both numbers are single digit
    if x.len() <= I and y.len() <= I:
      out_res = x.__mul(y)
    else:
      out_res = copy(E)
      carry = copy(O)
      mag = copy(E)
      # Outer loop for multiplying with each digit of y
      while y != O:
        fac = y[O]
        y = y >> I
        x_c = copy(x)
        in_res = copy(E)
        # Inner loop for multiplying
        while x_c != O:
          term = x_c[O]
          x_c = x_c >> I
          dm = fac.__mul(term)
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

  def __sub(x, y, vis=INVIS):
    return TInt(int(x) - int(y))

  def __sub__(x, y, vis=VIS):
    #print("SUBTRACT: ", x, y)
    assert x >= y
    res = copy(E)
    borrow = copy(O)
    while y != O:
      #print("looping...", x, y, borrow, res)
      # get next digis
      digx = x[O]
      digy = y[O]
      x = x >> I
      y = y >> I
      # factor in borrow
      #print("got digx, digy, borrow: ", digx, digy, borrow)
      if borrow > O:
        if digx > O:
          digx = digx.__sub(I)
          borrow = O
        else:
          digx = I | TInt(0)
          digx = digx.__sub(I)
      #print("facored in borrow: digx, borrow", digx, borrow)
      # subtract
      if digx < digy:
        assert borrow == O
        digx = I | digx
        borrow = I
      #print("borrowed as needed digx, borrow", digx, borrow)
      dd = digx.__sub(digy)
      res = dd | res
      #print("loop done. x, y, borrow, res:", x, y, borrow, res)
    if borrow != O:
      dd = x.__sub(I)
    else:
      dd = x
    if dd > O:
      res = dd | res
    return res

  def __floordiv__(x, y, vis=VIS):
    """
      Implements x // y using grade-school long division
    """
    res = copy(O)
    while x >= y:
      # choose factor
      #print("div x, y, res: ", x, y, res)
      len_x = x.len()
      len_y = y.len()
      # we know len_x >= len_y
      factor_mag = len_x - len_y
      if (y << factor_mag) > x:
        factor_mag = factor_mag - I
      #print(x, y, factor_mag)
      dig = I
      sum_div = y
      sub_x = x >> factor_mag
      # TODO make below process of dig guess better
      while sum_div <= sub_x:
        dig = dig + I
        sum_div = sum_div + y
        #print("iter", x, y, sum_div + y, sub_x)
      if sum_div > sub_x:
        dig = dig - I
        sum_div = sum_div - y
      #print("factor sum_div, dig: ", sum_div << factor_mag, dig)
      factor = dig << factor_mag
      # add factor to res
      res = res + factor
      remove = sum_div << factor_mag
      x = x - remove
    return res


O = TInt("0")
E = TInt("")
I = TInt("1")

if __name__ == "__main__":
  tests = [(324, 6), (6, 324), (199, 1), (199, 2), (500, 200), (970, 30), (907, 93), (9, 2), (0, 62), ("0023", "152")]
  #print("024 >> '0' = ", TInt("024") >> TInt("0"))
  for x, y in tests:
    x = TInt(x)
    y = TInt(y)
    print(x, y)
    print("x + y = ", x + y)
    if x >= y: print("x - y = ", x - y)
    else: print("y - x = ", y - x)
    print("x * y = ", x * y)
    print("x // y = ", x // y)

