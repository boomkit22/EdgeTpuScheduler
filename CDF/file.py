

f = open('array', 'r')
# print(f.readline())

lines = f.readlines()
array = []
for line in lines:
    array.append(float(line))

print(array)

# while f.readline():
    # print(f.read)
# x = [1,2,3]


# f = open('array','w')

# # f.write(x)

# # f.close()

# for e in x:
#     f.write(str(e) + "\n")

# f.close()