```Notations```

n - number of distinct data points
p - denote the number of variables that are available

e.q. ; Wage 
12 variables; 3000 people
n = 3000, p = 12

(12 represents; age, sex, year ...)

from the data points
xij ; note ij are subscripts

i = 1,2,...,n
j = 1,2,...,p

X is used when denoting [np] (the whole matrix)

e.q.; 

**X** = [x11  x12 ... x1p
         x21  x22 ... x2p
         .    .  .    .
         .    .    .  .
         .    .      ..
         xn1  xn2 ... xnp]


vectors are represented by a single column of values

e.q.;

xj = [  x1j
        x2j
        .
        .
        .
        xnj
            ]

e.q.;2 Wage data - year

**X** = [x1 x2  ... xp],

can be written as

**X** = [   x1T^
            x2T^
            .
            .
            .
            xnT^
                ]
['note: T^ is a <superscript> for transpose']

yi is to denote the ith observation of the variable
on which we wish to make predictions, such as wage.

y = [
     y1
     y2
     .
     .
     .
     yn
    ]

vector length n will always be denoted with a lower case bold

a = [
        a1
        a2
         .
         .
         .
         an
           ]

in the book
vectors that are not of length *n* will be denoted in 
lower normal font 

[∈ - is an element of]
a ∈ R ; a is an element of real numbers
a ∈ Rk^ (a ∈ Rn^) ; a is an element of real numbers of vector k(n)
A ∈ R(r × s)^ ; indicates that an object is a r × s matrix
**AB** ; denotes (A ∈ R), (B ∈ R)


Matrix Multiplication

(i,j)th element of AB is computed by multiplying the ith row of A, and the corresponding
element jth column of B.
(AB)ij = Σd^k=1 aik bkj. e.q.;

A = [1 2
     3 4]

B = [5 6
     7 8]


         [1 2   [5 6      [1 x 5 + 2 + 7   1 x 6 + 2 x 8               [19 22
AB =                                                       =
          3 4]   7 8]      3 x 5 + 4 x 7   3 x 6 + 4 x 8]               43 50]


this operation produces an r × s matrix. 

