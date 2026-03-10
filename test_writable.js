let obj = Object.create(null, {
  x: { value: 42, writable: false, enumerable: true, configurable: false }
});
obj.x = 100;
console.log('After assignment:', obj.x);
let keys = Object.keys(obj);
console.log('Keys:', keys.length);
