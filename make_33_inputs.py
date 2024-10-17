import numpy as np

np.random.seed(42)

num_ones = 11
starts = np.zeros((33, 22))

c = 0
while sum(starts[-1,:]) == 0:
  one_locs = np.random.choice(list(range(22)), size=num_ones, replace = False)
  start = np.zeros(22)
  start[one_locs] = 1
  exists = np.any(np.all(starts == start, axis=1))
  if not exists: #check that this start doesn't exist yet
    starts[c, :] = start
    c+=1

#check that they are rotationally different
for i1, s1 in enumerate(starts):
  for i2, s2 in enumerate(starts):
    s1 = list(s1)
    s2 = list(s2)
    concatenated = s1 + s1
    is_it = any(concatenated[i:i+len(s2)] == s2 for i in range(len(s1)))
    if is_it:
      if i1 != i2:
        print(i1, i2)

with open("33_inputs.txt", 'w') as f:
    np.savetxt(f, starts, newline=" ")