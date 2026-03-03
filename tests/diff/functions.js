// Regression test: functions (non-recursive, iterative)
function add(a, b) {
    return a + b;
}
print(add(3, 4));
print(add(0, 0));
print(add(1, 1));

function iterativeFactorial(n) {
    var result = 1;
    var i = 2;
    while (i <= n) {
        result = result * i;
        i = i + 1;
    }
    return result;
}
print(iterativeFactorial(5));
print(iterativeFactorial(1));

var square = function(x) {
    return x * x;
};
print(square(7));
