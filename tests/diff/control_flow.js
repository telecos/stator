// Regression test: variables and control flow
var x = 10;
var y = 0;
if (x > 5) {
    y = 1;
} else {
    y = 2;
}
print(y);

var sum = 0;
var i = 0;
while (i < 5) {
    sum = sum + i;
    i = i + 1;
}
print(sum);

var count = 0;
var j = 0;
for (; j < 3; j = j + 1) {
    count = count + 1;
}
print(count);
