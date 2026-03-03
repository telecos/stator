// Regression test: exception handling
var caught = false;
try {
    throw 42;
} catch (e) {
    caught = true;
    print(e);
}
print(caught);

function mayThrow(x) {
    if (x < 0) {
        throw "negative";
    }
    return x * 2;
}

try {
    print(mayThrow(5));
    print(mayThrow(0));
} catch (e) {
    print(e);
}
