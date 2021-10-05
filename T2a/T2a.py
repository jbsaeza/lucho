def calcular_max_long(seq):
    longest_seq = 0
    stack = [-1]
    for p in range(len(seq)):
        if seq[p] == "(":
            stack.append(p)
        else:
            stack.pop()
            if len(stack) == 0:
                stack.append(p)
            else:
                length = p-stack[-1]
                if length > longest_seq:
                    longest_seq = length
    return longest_seq


parentesis = "((()()"
max_long = calcular_max_long(parentesis)
print(max_long)