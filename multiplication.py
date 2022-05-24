def multiplicate(A):
    B = [0] * len(A)
    multiplication = 1
    number_of_zeros = 0
    non_zero_multiplication = 1
    for i, a in enumerate(A):
        if a != 0:
            multiplication *= a
        else:
            index_of_zero = i
            number_of_zeros += 1
    if number_of_zeros > 1:
        pass
    elif number_of_zeros == 1:
        B[index_of_zero] = multiplication
    else:
        for i, a in enumerate(A):
            B[i] = multiplication // a
    return B
        
        