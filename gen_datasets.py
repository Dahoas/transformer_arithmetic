import itertools

from func import *
from gen_arithmetic_data import *


if __name__ == "__main__":
    test_arithmetic = False
    test_sort = False
    if test_arithmetic:
        from inspect import signature
        TInt.update_vis("__add__", CALL)
        TInt.update_vis("__sub__", CALL)
        #TInt.set_dynamic_noise(0.1)
        x = torch.randint(0, 10, (6,)).tolist()
        y = torch.randint(0, 10, (3,)).tolist()
        x = TInt("".join([str(s) for s in x]))
        y = TInt("".join([str(s) for s in y]))
        #x = TInt(10719)
        #y = TInt(4623)
        #print(getattr(TInt, "__add__"))
        #print(signature(getattr(TInt, "__add__")))
        #z = x.__floordiv__(y)
        #print(z)
        TInt.update_vis("__add__", INVIS)
        TInt.update_vis("__sub__", INVIS)
        test_ops = ["gcd"]#, "sub", "mul", "div"]
        for op in test_ops:
            res = chain_of_thought_template(op,null_noise, x, y)
            resp = res["response"]
            print(res["prompt"] + resp)
            tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-410m")
            print("Pythia tok len: ", len(tok(resp).input_ids))
            tok = AutoTokenizer.from_pretrained("gpt2")
            print("gpt2 tok len: ", len(tok(resp).input_ids))
            print("\n")
        exit()
    if test_sort:
        include_code = False
        l = [199, 198, 197, 196, 195, 194, 193, 192, 191, 190, 189,188,187,186,185,184,183,182,181,180]
        tIntL = [TInt(x) for x in l]
        res = chain_of_thought_template("sort", null_noise, tIntL)
        resp = res["response"]
        print(res["prompt"] + resp)
        tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-410m")
        print("Pythia tok len: ", len(tok(resp).input_ids))
        tok = AutoTokenizer.from_pretrained("gpt2")
        print("gpt2 tok len: ", len(tok(resp).input_ids))
        print("\n")
        exit()


    num_train = 20000 // num_procs
    num_test = 1200

    dicts = []
    """for i in range(10):
        add_train_dict = {
            "num_samples": num_train,
            "arg_sampling": [["len", 1, 10], ["len", 1, 10]],
            "visibility": {},
            "dynamic_noise": 0.05 * i,
        }
        add_test_dict = {
          "num_samples": num_test,
          "arg_sampling": [["len", 1, 10], ["len", 1, 10]],
          "visibility": {},
          "dynamic_noise": 0,
        }
        dicts.append(({"add": add_train_dict}, {"add": add_test_dict}))

    mul_train_dict = {
      "num_samples": num_train,
      "arg_sampling": [["len", 1, 5], ["len", 1, 5]],
      "visibility": {"__add__": CALL},
    }
    mul_test_dict = {
      "num_samples": num_test,
      "arg_sampling": [["len", 1, 5], ["len", 1, 5]],
      "visibility": {"__add__": CALL},
    }"""

    gcd_train_dict = [{
      "num_samples": num_train,
      "arg_sampling": [["len", 1, 4], ["len", 1, 4]],
      "visibility": {"__add__": INVIS,
                    "__sub__": INVIS},
    }]
    gcd_test_dict = [{
      "num_samples": num_test,
      "arg_sampling": [["len", 1, 4], ["len", 1, 4]],
      "visibility": {"__add__": INVIS,
                    "__sub__": INVIS},
    }]
    
    """lens = [x+5 for x in range(25)]
    sort_train = {"sort": []}
    sort_test = {"sort": []}
    for x in lens:
      sort_train_dict = {
       "num_samples": num_train // len(lens),
        "arg_sampling": [["len", 2, 10] for i in range(x)],
        "visibility": {},
      }
      sort_test_dict = {
        "num_samples": num_test // len(lens),
        "arg_sampling": [["len", 2, 10] for i in range(x)],
        "visibility": {},
      }
      sort_train["sort"].append( sort_train_dict)
      sort_test["sort"].append( sort_test_dict)

    dicts.append((sort_train, sort_test))"""
    #dicts.append(({"add": add_train_dict}, {"add": add_test_dict}))
    #dicts.append(({"mul": mul_train_dict}, {"mul": mul_test_dict}))
    dicts.append(({"gcd": gcd_train_dict}, {"gcd": gcd_test_dict}))

    """mix_train_dict = {
                            "add": {
                                        "num_samples": num_train // 4,
                                        "arg_sampling": [["len", 1, 10], ["len", 1, 10]],
                                        "visibility": {},
                                   },
                            "sub": {
                                        "num_samples": num_train // 4,
                                        "arg_sampling": [["len", 1, 10], ["len", 1, 10]],
                                        "visibility": {},
                                   },
                            "mul": {
                                        "num_samples": num_train // 4,
                                        "arg_sampling": [["len", 1, 5], ["len", 1, 5]],
                                        "visibility": {
                                                        "__add__": CALL,
                                                      },
                                   },
                            "div": {
                                        "num_samples": num_train // 4,
                                        "arg_sampling": [["len", 1, 5], ["len", 1, 5, 1]],
                                        "visibility": {
                                                        "__add__": CALL,
                                                        "__sub__": CALL,
                                                      },
                                   },
                          }
    mix_test_dict = {
                            "add": {
                                        "num_samples": num_test // 4,
                                        "arg_sampling": [["len", 1, 10], ["len", 1, 10]],
                                        "visibility": {},
                                   },
                            "sub": {
                                        "num_samples": num_test // 4,
                                        "arg_sampling": [["len", 1, 10], ["len", 1, 10]],
                                        "visibility": {},
                                   },
                            "mul": {
                                        "num_samples": num_test // 4,
                                        "arg_sampling": [["len", 1, 5], ["len", 1, 5]],
                                        "visibility": {
                                                        "__add__": CALL,
                                                      },
                                   },
                            "div": {
                                        "num_samples": num_test // 4,
                                        "arg_sampling": [["len", 1, 5], ["len", 1, 5, 1]],
                                        "visibility": {
                                                        "__add__": CALL,
                                                        "__sub__": CALL,
                                                      },
                                   },
                          }"""
    #dicts.append((mix_train_dict, mix_test_dict))

    """dicts = []
    for i in range(3):
        add_train_dict = {
        "num_samples": num_train,
        "arg_sampling": [["len", 1, 10], ["len", 1, 10]],
        "linenoise": 0.05 * i + .9,
        "visibility": {},
        }
        add_test_dict = {
        "num_samples": num_test,
        "arg_sampling": [["len", 1, 10], ["len", 1, 10]],
        "visibility": {},
        }
        dicts.append(({"add": add_train_dict}, {"add": add_test_dict}))"""

    dicts = []

    lens = [x+4 for x in range(5)]
    sort_train = {"median": []}
    sort_test = {"median": []}
    for x in lens:
      sort_train_dict = {
       "num_samples": num_train // len(lens),
        "arg_sampling": [["len", 2, 8] for i in range(x)],
        "visibility": {"__floordiv__": CALL,
                       "__add__": CALL,
                       "__sub__": CALL,
                       "__mod__": CALL},
      }
      sort_test_dict = {
        "num_samples": num_test // len(lens),
        "arg_sampling": [["len", 2, 8] for i in range(x)],
        "visibility": {"__floordiv__": CALL,
                       "__add__": CALL,
                       "__sub__": CALL,
                       "__mod__": CALL},
      }
      sort_train["median"].append(sort_train_dict)
      sort_test["median"].append(sort_test_dict)

    dicts.append((sort_train, sort_test))

    prompt_templates = [chain_of_thought_template, no_template]

    d_noises = [1.0]
    char_noises = [0.25, 0.5, 0.75, 1.0]

    for prompt_template in prompt_templates:
        for doc_noise, char_noise in itertools.product(d_noises, char_noises):
            for train_samp, test_samp in dicts:
                # For now just noising add
                #if len(train_samp) == 1 and "add" in train_samp:
                gen_noisy_dataset(prompt_template, doc_noise, null_noise, char_noise, train_samp, test_samp)