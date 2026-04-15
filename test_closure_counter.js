function make_counter() {
  var count = 0;
  return function() { count = count + 1; return count; };
}
var c = make_counter();
var result = 0;
for (var i = 0; i < 10; i++) result = c();
return result;
