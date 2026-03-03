// Function declarations and calls.
function add(a, b) {
  return a + b;
}

function factorial(n) {
  if (n <= 1) return 1;
  return n * factorial(n - 1);
}

print(add(3, 4));
print(factorial(5));
print(factorial(10));
