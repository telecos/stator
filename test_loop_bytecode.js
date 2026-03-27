var n = 0;
for (var i = 0; i < 10000; i++) {
    n = (n + i * 3 - 1) | 0;
}
n;
