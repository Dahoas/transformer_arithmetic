from primitive_types import *
from primitive_ops import *

f = open("test.py")
lines = f.read().splitlines()

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

prev_vars = {}
prev_changed_var = ''
def custom_trace(frame, event, arg = None):
  global prev_vars, prev_changed_var
  #print(event, frame.f_lineno, frame.f_code, frame.f_locals)
  line_no = frame.f_lineno
  code_line = lines[line_no - 1].strip()
  local_vars = frame.f_locals
  #print(prev_vars, local_vars)
  relevant_vars = {k:v for (k,v) in local_vars.items() if k not in prev_vars or not prev_vars[k] == local_vars[k] or k == prev_changed_var}
  #print(relevant_vars)
  prev_changed_var = code_line.split("=")[0].strip()
  prev_vars = local_vars.copy()
  if len(relevant_vars) > 0:
    print(", ".join([str(k) + " = " + str(v) for (k, v) in relevant_vars.items()]))
  print(code_line)
  return custom_trace

def test_trace(l):
  import sys
  import trace

  #tracer = trace.Trace()
  #tracer.runfunc(addition, '123', '1234')

  #r = tracer.results()
  #r.write_results(show_missing=True, coverdir='.')
  sys.settrace(custom_trace)
  #ret = addition('123', '1234')
  ret = sort_integers(l)
  sys.settrace(None)
  return ret


if __name__ == '__main__':
    #FSimpleAdd = TSimpleAdd()
    #FTotalAdd = TAddition()

    #b0 = TVar(val = TInt(52))
    #b1 = TVar(val =TInt(59))
    #answer, trace = FTotalAdd.compute(b0, b1)
    #print(answer)
    #print(trace)
    l = ['4', '3', '2']
    test_trace(l)
    #pdb.set_trace()

