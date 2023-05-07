def mean(nums):
    total = 0
    count = 0
    for num in nums:
        total += num
        count += 1
    return total / count

def max(nums):
    result = nums[0]
    for num in nums:
        if num > result:
            result = num
    return result

def min(nums):
    result = nums[0]
    for num in nums:
        if num < result:
            result = num
    return result

def count(num, nums):
    result = 0
    for n in nums:
        if n == num:
            result += 1
    return result

def primes(n):
    primes = []
    for i in range(2, n+1):
        is_prime = True
        for j in range(2, int(i**0.5)+1):
            if i % j == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(i)
    return primes

def lcm(nums):
    def gcd_two(a, b):
        if b == 0:
            return a
        else:
            return gcd_two(b, a % b)
    result = nums[0]
    for num in nums[1:]:
        result = result * num // gcd_two(result, num)
    return result

def gcd(nums):
    def gcd_two(a, b):
        if b == 0:
            return a
        else:
            return gcd_two(b, a % b)
    result = nums[0]
    for num in nums[1:]:
        result = gcd_two(result, num)
    return result

def difference(nums1, nums2):
    result = []
    for i in range(len(nums1)):
        result.append(abs(nums1[i] - nums2[i]))
    return result

def cumulative_sum(nums):
    result = []
    total = 0
    for num in nums:
        total += num
        result.append(total)
    return result

def mean_squared_error(actual, predicted):
    total = 0
    count = 0
    for i in range(len(actual)):
        total += (actual[i] - predicted[i]) ** 2
        count += 1
    return total / count

def count(num, nums):
    result = 0
    for n in nums:
        if n == num:
            result += 1
    return result

def product(nums):
    result = 1
    for num in nums:
        result *= num
    return result

def gcd(a, b):
    if b == 0:
        return a
    else:
        return gcd(b, a % b)

def factorial(n):
    if n < 0:
        raise ValueError("Factorial is not defined for negative integers")
    result = 1
    for i in range(1, n+1):
        result *= i
    return result

def variance(nums):
    m = mean(nums)
    total = 0
    count = 0
    for num in nums:
        total += (num - m) ** 2
        count += 1
    return total / (count - 1)

def range(nums):
    return max(nums) - min(nums)

def median(nums):
    nums_sorted = sorted(nums)
    n = len(nums_sorted)
    if n % 2 == 0:
        return (nums_sorted[n // 2 - 1] + nums_sorted[n // 2]) / 2
    else:
        return nums_sorted[n // 2]

def even_odd(nums):
    evens = []
    odds = []
    for num in nums:
        if num % 2 == 0:
            evens.append(num)
        else:
            odds.append(num)
    return (evens, odds)

def unique(nums):
    result = []
    for num in nums:
        if num not in result:
            result.append(num)
    return result

def reverse(nums):
    result = nums[:]
    n = len(result)
    for i in range(n // 2):
        result[i], result[n-1-i] = result[n-1-i], result[i]
    return result

def sorted(nums):
    result = nums[:]
    n = len(result)
    for i in range(n):
        for j in range(i+1, n):
            if result[j] < result[i]:
                result[i], result[j] = result[j], result[i]
    return result

def sum_squares(nums):
    result = 0
    for num in nums:
        result += num * num
    return result

def prime_factors(n):
    factors = []
    d = 2
    while n > 1:
        while n % d == 0:
            factors.append(d)
            n //= d
        d += 1
        if d*d > n:
            if n > 1:
                factors.append(n)
            break
    return factors

def power_of_two(n):
    if n < 1:
        return False
    while n % 2 == 0:
        n //= 2
    return n == 1

def binary_search(nums, target):
    low, high = 0, len(nums)-1
    while low <= high:
        mid = (low + high) // 2
        if nums[mid] == target:
            return mid
        elif nums[mid] < target:
            low = mid + 1
        else:
            high = mid - 1
    return -1

def is_palindrome(nums):
    n = len(nums)
    for i in range(n // 2):
        if nums[i] != nums[n-1-i]:
            return False
    return True

def rotate_left(nums, k):
    n = len(nums)
    k %= n
    for i in range(k):
        temp = nums[0]
        for j in range(n-1):
            nums[j] = nums[j+1]
        nums[n-1] = temp
    return nums

def fibonacci(n):
    if n <= 1:
        return n
    a, b = 0, 1
    for i in range(2, n+1):
        a, b = b, a+b
    return b

def lcm(a, b):
    return abs(a * b) // gcd(a, b)

def decimal_to_binary(decimal):
    binary = ''
    while decimal > 0:
        binary = str(decimal % 2) + binary
        decimal = decimal // 2
    return binary

def digit_sum(n):
    result = 0
    while n > 0:
        result += n % 10
        n //= 10
    return result

def is_armstrong(n):
    k = len(str(n))
    temp = n
    result = 0
    while temp > 0:
        digit = temp % 10
        result += digit**k
        temp //= 10
    return result == n

def is_happy(n):
    seen = set()
    while n not in seen:
        seen.add(n)
        n = sum(int(digit)**2 for digit in str(n))
        if n == 1:
            return True
    return False

def is_automorphic(n):
    square = n**2
    while n > 0:
        if n % 10 != square % 10:
            return False
        n //= 10
        square //= 10
    return True

def is_circular_prime(n):
    if not is_prime(n):
        return False
    num_str = str(n)
    for i in range(len(num_str)):
        rotated_num_str = num_str[i:] + num_str[:i]
        if not is_prime(int(rotated_num_str)):
            return False
    return True

# NOTE: Need to adapt str for TInt
# Also need to make __iter__ class
def is_harshad(n):
    digit_sum = sum(int(digit) for digit in str(n))
    return n % digit_sum == 0

def is_kaprekar(n):
    square = n**2
    num_digits = len(str(n))
    right_half = square % 10**num_digits
    left_half = square // 10**num_digits
    return n == left_half + right_half

def is_lucky(n):
    sieve = list(range(1, n+1))
    i = 1
    while sieve[i] <= n:
        j = sieve[i]
        while j < n:
            sieve.remove(j)
            j += sieve[i]
        i += 1
    return n in sieve

def is_pandigital(n):
    num_str = str(n)
    if len(num_str) != 10:
        return False
    return all(str(digit) in num_str for digit in range(10))

def divisors(n):
    divs = []
    for i in range(1, int(n**0.5)+1):
        if n % i == 0:
            divs.append(i)
            if i != n // i:
                divs.append(n // i)
    divs.sort()
    return divs


def is_abundant(n):
    return sum(divisors(n)[:-1]) > n

def is_deficient(n):
    return sum(divisors(n)[:-1]) < n

def is_even(n):
    return n % 2 == 0

def is_mersenne_prime(n):
    if not is_prime(n):
        return False
    i = 1
    while 2**i - 1 < n:
        if 2**i - 1 == n:
            return True
        i += 1
    return False

def is_odd(n):
    return n % 2 == 1

def is_perfect_power(n):
    for i in range(2, int(n**0.5)+1):
        j = i
        while j <= n:
            j *= i
            if j == n:
                return True
    return False

def is_pronic(n):
    for i in range(int(n**0.5)+1):
        if i * (i+1) == n:
            return True
    return False

def is_square_free(n):
    for i in range(2, int(n**0.5)+1):
        if i**2 > n:
            break
        if n % i**2 == 0:
            return False
    return True

def is_triangular(n):
    i = 1
    while n > 0:
        n -= i
        i += 1
    return n == 0

def is_pentagonal(n):
    i = 1
    while n > 0:
        n -= 3*i - 2
        i += 1
    return n == 0

def is_prime_power(n):
    if n == 1:
        return True
    for p in primes(n):
        if n % p == 0:
            exp = 1
            while n % p**exp == 0:
                exp += 1
            return n == p**(exp-1)
    return False

def factorize(n):
    factors = []
    i = 2
    while i*i <= n:
        if n % i == 0:
            factors.append(i)
            n //= i
        else:
            i += 1
    if n > 1:
        factors.append(n)
    return factors


def euler_phi(n):
    phi = n
    for p in set(factorize(n)):
        phi //= p
        phi *= p-1
    return phi

def is_square_pyramid(n):
    i = 1
    while n > 0:
        n -= i**2
        i += 1
    return n == 0

# NOTE: Need to be able to multiply TInts by ints
def is_square_triangle(n):
    return is_square(8*n + 1)


def is_elysian(n):
    squares = []
    i = 1
    while i**2 <= n:
        squares.append(i**2)
        i += 1
    for a in squares:
        for b in squares:
            if a + b == n and a != b:
                return True
    return False

def is_lychrel(n):
    for i in range(50):
        n += int(str(n)[::-1])
        if str(n) == str(n)[::-1]:
            return False
    return True

def sum(nums):
    res = 0
    for i in range(len(nums)):
        res += nums[i]
    return nums

def is_perfect(n):
    divisors = [d for d in range(1, n) if n % d == 0]
    return n == sum(divisors)

def is_strong(n):
    digits = [int(d) for d in str(n)]
    factorials = [factorial(d) for d in digits]
    digit_factorials_sum = sum(factorials)
    return n == digit_factorials_sum

def is_tribonacci(n):
    a, b, c = 0, 0, 1
    while c < n:
        a, b, c = b, c, a + b + c
    return c == n

def is_b_smooth(n, b):
    return max(prime_factors(n)) <= b

def is_hcn(n):
    i = 1
    while (3*i**2 - i) < n:
        i += 1
    return (3*i**2 - i) == n

def is_hypotenuse(n):
    for a in range(1, int(n**0.5)):
        b = int((n - a**2)**0.5)
        if a**2 + b**2 == n:
            return True
    return False

def is_kempner(n):
    for i in range(1, len(str(n))+1):
        if n % factorial(i) == 0:
            return False
    return True

def mode(nums):
    counts = {}
    for x in nums:
        counts[x] = counts.get(x, 0) + 1
    max_count = max(counts.values())
    modes = [k for k, v in counts.items() if v == max_count]
    return modes[0] if modes else None

def variance(nums):
    count = len(nums)
    if count <= 1:
        return 0
    else:
        total = sum(nums)
        mean = total / count
        deviations = [(x - mean)**2 for x in nums]
        variance = sum(deviations) / (count - 1)
        return variance


# NOTE: Need to convert division to floordiv. Or just overload to floor div
def standard_deviation(nums):
    return variance(nums) ** 0.5

def skewness(nums):
    n = len(nums)
    if n < 3:
        return 0
    else:
        m = mean(nums)
        s = standard_deviation(nums)
        numerator = sum((x - m)**3 for x in nums)
        denominator = (n-1) * (n-2) * s**3
        return numerator / denominator

def kurtosis(nums):
    n = len(nums)
    if n < 4:
        return 0
    else:
        m = mean(nums)
        s = standard_deviation(nums)
        numerator = sum((x - m)**4 for x in nums)
        denominator = (n-1) * (n-2) * (n-3) * s**4
        excess = (numerator / denominator) - 3
        return excess

def covariance(x, y):
    n = len(x)
    if n < 2:
        return 0
    else:
        m_x, m_y = mean(x), mean(y)
        deviations_x = [xi - m_x for xi in x]
        deviations_y = [yi - m_y for yi in y]
        covariance = sum(xi * yi for xi, yi in zip(deviations_x, deviations_y)) / (n - 1)
        return covariance

def interquartile_range(nums):
    sorted_nums = sorted(nums)
    n = len(sorted_nums)
    q1 = median(sorted_nums[:n//2])
    q3 = median(sorted_nums[(n+1)//2:])
    return q3 - q1

def z_score(x, nums):
    m = mean(nums)
    s = standard_deviation(nums)
    if s == 0:
        return 0
    else:
        return (x - m) / s

def binomial_coefficient(n, k):
    if k < 0 or k > n:
        return 0
    else:
        k = min(k, n - k)
        c = 1
        for i in range(k):
            c *= n - i
            c //= i + 1
        return c

def midrange(nums):
    if len(nums) > 0:
        return (max(nums) + min(nums)) / 2
    else:
        return 0

def mad(nums):
    median_val = median(nums)
    deviations = [abs(x - median_val) for x in nums]
    return median(deviations)

def moment(nums, k):
    m = mean(nums)
    return sum((x - m)**k for x in nums) / len(nums)

def catalan(n):
    if n <= 1:
        return 1
    else:
        c = 1
        for i in range(n):
            c *= (4*i+2) / (i+2)
        return int(c)

def stirling_first(n, k):
    if n == k == 0:
        return 1
    elif n == 0 or k == 0:
        return 0
    else:
        s = [[0]*(n+1) for i in range(k+1)]
        s[0][0] = 1
        for i in range(1, k+1):
            for j in range(1, n+1):
                s[i][j] = (j-1) * s[i][j-1] + s[i-1][j-1]
        return s[k][n]

def stirling_second(n, k):
    if n == k == 0:
        return 1
    elif n == 0 or k == 0:
        return 0
    else:
        S = [[0]*(n+1) for i in range(k+1)]
        S[0][0] = 1
        for i in range(1, k+1):
            for j in range(1, n+1):
                S[i][j] = i*S[i][j-1] + S[i-1][j-1]
        return S[k][n]

def bell_number(n):
    B = [[0]*(n+1) for i in range(n+1)]
    B[0][0] = 1
    for i in range(1, n+1):
        B[i][0] = B[i-1][i-1]
        for j in range(1, i+1):
            B[i][j] = B[i-1][j-1] + B[i][j-1]
    return B[n][0]

def eulerian_number(n, m):
    if m == 0:
        return int(n == 0)
    elif m == 1:
        return int(n * (n-1) // 2)
    else:
        A = [[0]*(m+1) for i in range(n+1)]
        for i in range(n+1):
            A[i][0] = 1
            A[i][1] = i
        for i in range(1, n+1):
            for j in range(2, m+1):
                A[i][j] = (j-1) * A[i-1][j] + (i-j+1) * A[i-1][j-1]
        return A[n][m]

def eratosthenes_sieve(n):
    # Create a list of booleans representing whether each integer is prime or not
    is_prime = [True] * (n+1)
    is_prime[0], is_prime[1] = False, False

    # Use the Sieve of Eratosthenes to mark all non-prime integers
    for i in range(2, int(n**0.5)+1):
        if is_prime[i]:
            for j in range(i**2, n+1, i):
                is_prime[j] = False

    # Return a list of all prime integers less than or equal to n
    return [i for i in range(n+1) if is_prime[i]]

def derangements(n):
    # Raise a ValueError if n is negative
    if n < 0:
        raise ValueError('n cannot be negative')

    # Use the formula for the number of derangements of n elements
    result = 1
    for i in range(2, n+1):
        result *= i
        if i % 2 == 0:
            result += 1
        else:
            result -= 1

    return result

def motzkin_number(n):
    # Raise a ValueError if n is negative
    if n < 0:
        raise ValueError('n cannot be negative')

    # Use a loop to calculate the nth Motzkin number
    M = [1, 1]
    for i in range(2, n+1):
        M_i = M[i-1]
        for j in range(1, i):
            M_i += M[j-1] * M[i-j-1]
        M.append(M_i)

    return M[n]

def domino_tiling(m, n):
    # Use a loop to calculate the number of ways to tile an m x n grid with dominoes
    d = [1, 1, 2]
    for i in range(3, m+1):
        d_i = d[i-1] + d[i-2]
        for j in range(3, n+1):
            d_i += 2 * d[i-1] - d[i-3]
        d.append(d_i)

    return d[m]

def quadrant_partitions(n):
    # Raise a ValueError if n is negative
    if n < 0:
        raise ValueError('n cannot be negative')

    # Use a loop to calculate the number of quadrant partitions of n
    partitions = [0] * (n+1)
    partitions[0] = 1
    for i in range(1, n+1):
        for j in range(i, n+1):
            partitions[j] += partitions[j-i]

    return partitions[n]

def kneser_table(n, k):
    # Raise a ValueError if k is negative or greater than n
    if k < 0 or k > n:
        raise ValueError('k must be between 0 and n')

    # Use a loop to calculate the entries of the Kneser table
    table = [[0]*(k+1) for i in range(n+1)]
    for i in range(n+1):
        for j in range(min(i, k)+1):
            if j == 0:
                table[i][j] = 1
            else:
                table[i][j] = table[i-1][j-1] + table[i-j][j]

    return table

def young_tableau_partitions(shape):
    # Raise a ValueError if shape is not a Young diagram
    if not is_young_diagram(shape):
        raise ValueError('shape is not a valid Young diagram')

    # Use a loop to calculate the number of standard Young tableaux of the given shape
    n = sum(shape)
    partitions = [0] * (n+1)
    partitions[0] = 1
    for i in range(1, n+1):
        for j in range(i, n+1):
            partitions[j] += partitions[j-i]

    return partitions
