import time
def logged(func):
    def logged_decorator(*args, **kwargs):
        res = func(*args, **kwargs)
        print('you called',func.__name__,args)
        print('it returned' ,res)
    return logged_decorator
@logged
def func(*args, **kwargs):
    return 3+len(args)

func(4,4,4)