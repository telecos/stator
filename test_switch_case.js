var log = '';
function tag(n) { 
  log = log + n; 
  return n; 
}
var x = 2;
switch(x) {
  case tag(1): break;
  case tag(2): log = log + 'hit'; break;
  case tag(3): break;
}
console.log('log =', log);
console.log('Expected: 12hit');
console.log('Match:', log === '12hit');
