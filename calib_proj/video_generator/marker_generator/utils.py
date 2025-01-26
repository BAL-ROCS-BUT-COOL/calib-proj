
def closest_multiple(x, n):
    # Compute the quotient (integer division)
    quotient = x // n
    
    # Compute the two nearest multiples
    lower_multiple = n * quotient
    upper_multiple = n * (quotient + 1)
    
    # Compare which multiple is closer to x
    if abs(x - lower_multiple) <= abs(x - upper_multiple):
        return lower_multiple
    else:
        return upper_multiple
    

def is_multiple(x, n):
    return x % n == 0