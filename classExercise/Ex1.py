Rot13 = {'a':'n', 'b':'o',
         'c':'p', 'd':'q', 'e':'r', 'f':'s', 'g':'t', 'h':'u', 'i':'v', 'j':'w',
         'k':'x', 'l':'y', 'm':'z', 'n':'a', 'o':'b', 'p':'c', 'q':'d', 'r':'e',
         's':'f', 't':'g', 'u':'h', 'v':'i', 'w':'j', 'x':'k','y':'l', 'z':'m',
         'A':'N','B':'O', 'C':'P', 'D':'Q', 'E':'R', 'F':'S', 'G':'T', 'H':'U',
         'I':'V', 'J':'W', 'K':'X', 'L':'Y', 'M':'Z', 'N':'A', 'O':'B', 'P':'C',
         'Q':'D', 'R':'E', 'S':'F', 'T':'G', 'U':'H', 'V':'I', 'W':'J', 'X':'K',
         'Y':'L', 'Z':'M'}
# #encoder
str1 = input("enter your secret message: ")
temp1 = ''
for x in str1:
    if(x in Rot13.values()):
        temp1 +=Rot13[x]
else:
    pass
print(temp1)
# ####################################################################################
#decoder
str2 = input("please enter the encoding text to Decipher the mystery: ")
temp2 = ''
listOfItems = Rot13.items()
for x in str2:
    for item in listOfItems:
        if item[1] == x:
            temp2 += item[0]
    else:
        pass
print(temp2)
####################################################################################