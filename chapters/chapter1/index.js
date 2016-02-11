var math = require('forwardjs-ml-math');

var data = [{
  input: [0, 0],
  output: 0,
},{
  input: [0, 1],
  output: 1,
},{
  input: [1, 0],
  output: 1,
},{
  input: [1, 1],
  output: 1,
}];

// Initial parameters.
var a = 3, b = -4, c = 2;

var z;
function and(x, y) {
  z = a + b*x + c*y;
  return math.sigmoid(z);
}

var STEP_SIZE = 0.1;

for (var iter = 0; iter < 10000; iter++) {
  // pick a random data point
  var i = Math.floor(Math.random() * data.length);
  var input = data[i].input;
  var x = input[0];
  var y = input[1];

  var output = data[i].output;

  var h = and(x, y);

  var error = h - output;

  a -= error * 1 * math.sigmoidGradient(z) * STEP_SIZE;
  b -= error * x * math.sigmoidGradient(z) * STEP_SIZE;
  c -= error * y * math.sigmoidGradient(z) * STEP_SIZE;

  if (iter % 25 === 0) console.log('accuracy at iter %s: %s', iter, accuracy());
}

console.log('a', a);
console.log('b', b);
console.log('c', c);

function accuracy() {
  var correct = 0;
  for (var i = 0; i < data.length; i++) {
    var input = data[i].input;
    var x = input[0];
    var y = input[1];

    var output = data[i].output;

    var h = and(x, y) > 0.5 ? 1 : 0;
    if (h === output) correct++;
  }
  return correct / data.length;
}
