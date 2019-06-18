s=0
a=[int(i) for i in input().split()]
for i in a:
    s+=i
    if(s>=0):
        print(i)
