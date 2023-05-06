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

  def __init__(self, val, vis=False):
    """
    Remove leading 0s and store in val
    Note "" is 0 is 00000
    """
    self.val = str(val)#re.sub(r"^00*", "", str(val))

  def __repr__(self):
    return str(int(self))
    #return "Val: {}, Int: {}".format(self.val, int(self))

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
    if self > O: return int(self.val)
    else: return 0

  def __eq__(x, y, vis=False):
    return x.val == y.val

  def __gt__(x, y, vis=False):
    return x.val > y.val

  def __or__(x, y, vis=False):
    """
    Implements TInt concatenation
    """
    return TInt(x.val + y.val)

  def __is_zero(self):
    return self.val == ""

  def __getitem__(self, ind, vis=False):
    if self.val == "": return copy(O)
    print(self.val, self.len(), ind)
    if not (self.len() > ind):
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
    assert not x.len() > I
    assert not y.len() > I
    return TInt(int(x) + int(y))

  def __add__(x, y, vis=True):
    #print("ADD: ", x, y)
    if x == O: return copy(y)
    if y == O: return copy(x)
    if not (x.len() > I or y.len() > I):
      return x.__add(y)
    #print("basecase not returned...")
    res = copy(O)
    carry = copy(O)
    while x > O or y > O:
      #print("preiter: ds, x, y, carry: ", res, x, y, carry)
      dig1 = x[O]
      dig2 = y[O]
      ds = dig1 + dig2
      ds = ds + carry
      res = ds[O] | res
      ds = ds.drop()
      carry = copy(ds)
      x = x.drop()
      y = y.drop()
      #print("postiter: carry, ds, res", carry, ds, res)
    if ds > O:
      res = ds | res
    return res

O = TInt("")
I = TInt("1")

x = TInt(155)
y = TInt(99999)
z = x + y
print(z)
