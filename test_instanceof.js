try {
  throw new TypeError("test");
} catch (e) {
  console.log("Error instance:", e);
  console.log("Is TypeError?", e instanceof TypeError);
}
