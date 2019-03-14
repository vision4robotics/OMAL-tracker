function flag = istriangle(a, b, c)

if a <= 0 || b <= 0 || c <= 0 || ( a + b <= c ) || ( b + c <= a ) || ( c + a <= b )
    flag = false; 
else
    flag = true; 
end