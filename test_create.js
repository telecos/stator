// Test Object.create with property descriptors
let obj = Object.create(null, {
  x: { value: 42, writable: false, enumerable: true, configurable: false },
  y: { value: 99, writable: true, enumerable: false, configurable: true }
});
console.log(obj.x);
console.log(obj.y);
console.log(Object.getOwnPropertyDescriptor(obj, 'x'));
console.log(Object.getOwnPropertyDescriptor(obj, 'y'));
