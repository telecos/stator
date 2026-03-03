// Regression test: higher-order functions (non-capturing)
// Note: free variable capture across function scope boundaries is not
// yet implemented; these tests use only argument passing.
function applyTwice(f, x) {
    return f(f(x));
}
function double(n) {
    return n * 2;
}
print(applyTwice(double, 3));
print(applyTwice(double, 1));

function applyThree(f, x) {
    return f(f(f(x)));
}
function addOne(n) {
    return n + 1;
}
print(applyThree(addOne, 10));
