from copy import copy
import re
from enum import Enum
import functools
import numpy as np


"""
Things to remove in processing
1. Functions with bad visibility
2. copy of TInt
3. base function names
"""


INVIS = 0  # Invis functions generate no trace
VIS = 1  # Visible functions generate a trace
CALL = 2  # Called functions are wrapped in call(...) with no trace


class TInt:
  val = "0"
  noise_pr = 0
  # Dict tracking list of functions to make visible in trace
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
              "__getitem__": INVIS,
              "__add__": VIS,
              "__rshift__": INVIS,
              "__lshift__": INVIS,
              "__mul__": VIS,
              "__sub__": VIS,
              "__floordiv__": VIS,
            }

  def __init__(self, val, vis=INVIS):
    """
    Remove leading 0s and store in val
    Note "" is 0 is 00000
    """
    self.val = str(val)
    # noise
    if self.noise_pr == 0: return
    
    samples = np.random.rand(len(self.val))
    randoms = np.random.randint(0, 9, len(self.val))
    corrupted_string = ''
    found_nonzero = False
    for i in range(len(self.val)):
      if int(self.val[i]) > 0: found_nonzero = True
      if found_nonzero: 
        if samples[i] < self.noise_pr: corrupted_string += str(randoms[i])
        else: corrupted_string += str(self.val[i])
      else: corrupted_string += str(self.val[i])
    self.val = corrupted_string

  @classmethod
  def update_vis(cls, f_name, vis):
    """Updates default visibility of input function f_name
    """
    setattr(cls, f_name, functools.partialmethod(getattr(TInt, f_name), vis=vis))

  @classmethod
  def reset_vis(cls):
    for f_name in cls.VIS_DICT:
      cls.update_vis(f_name, cls.VIS_DICT[f_name])

  @classmethod
  def set_dynamic_noise(cls, p):
    cls.noise_pr = p

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
    assert x.len() <= I or TInt.noise_pr > 0
    assert y.len() <= I or TInt.noise_pr > 0
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
    assert x >= y or TInt.noise_pr > 0
    return TInt(np.abs(int(x) - int(y)))

  def __sub__(x, y, vis=VIS):
    assert x >= y or TInt.noise_pr > 0
    res = copy(E)
    borrow = copy(O)
    while y != O:
      # get next digis
      digx = x[O]
      digy = y[O]
      x = x >> I
      y = y >> I
      # factor in borrow
      if borrow > O:
        if digx > O:
          digx = digx.__sub(I)
          borrow = O
        else:
          digx = I | TInt(0)
          digx = digx.__sub(I)
      # subtract
      if digx < digy:
        assert borrow == O or TInt.noise_pr > 0
        digx = I | digx
        borrow = I
      dd = digx.__sub(digy)
      res = dd | res
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
      len_x = x.len()
      len_y = y.len()
      # we know len_x >= len_y
      factor_mag = len_x - len_y
      if (y << factor_mag) > x:
        factor_mag = factor_mag - I
      dig = I
      sum_div = y
      sub_x = x >> factor_mag
      # TODO make below process of dig guess better
      while sum_div <= sub_x:
        dig = dig + I
        sum_div = sum_div + y
      if sum_div > sub_x:
        dig = dig - I
        sum_div = sum_div - y
      factor = dig << factor_mag
      # add factor to res
      res = res + factor
      remove = sum_div << factor_mag
      x = x - remove
    return res

  def __mod__(x, y, vis=VIS):
    """
      Implements x // y using grade-school long division
    """
    while x >= y:
      # choose factor
      len_x = x.len()
      len_y = y.len()
      # we know len_x >= len_y
      factor_mag = len_x - len_y
      if (y << factor_mag) > x:
        factor_mag = factor_mag - I
      sum_div = y
      sub_x = x >> factor_mag
      # TODO make below process of dig guess better
      while sum_div <= sub_x:
        sum_div = sum_div + y
      if sum_div > sub_x:
        sum_div = sum_div - y
      # add factor to res
      remove = sum_div << factor_mag
      x = x - remove
    return x


######## Constants ########

O = TInt("0")
E = TInt("")
I = TInt("1")


######## TInt functions ########

def sort_ints(l, vis=VIS):
  sorted_l = []
  while len(l) > 0:
    # find min selection sort style
    min_item = l[0]
    min_index = 0
    for i in range(len(l)):
      candidate = l[i]
      if candidate < min_item:
        min_item = candidate
        min_index = i
    sorted_l.append(min_item)
    l = l[:min_index] + l[min_index+1:]
  return sorted_l


def euclidean_alg(x, y, vis=VIS):
  """while x != y:
    if x > y: 
      x = x - y
    else:
      y = b - x
  return x"""
  while y != O:
    t = y
    y = x % y
    x = t
  return x


def median(nums, vis=VIS):
  nums_sorted = sort_ints(nums)
  n = len(nums_sorted)
  if n % 2 == 0:
    # average mids
    left_mid = nums_sorted[n // 2 - 1]
    right_mid = nums_sorted[n // 2]
    sum_mid = left_mid + right_mid
    res = sum_mid // TInt(2)
  else:
    res = nums_sorted[n // 2]
  return res