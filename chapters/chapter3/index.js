var math = require('forwardjs-ml-math');
var Network = require('./network.js');
// var Network = require('./network.es6');

var data = [{
  input: [0, 0],
  output: [0],
},{
  input: [0, 1],
  output: [1],
},{
  input: [1, 0],
  output: [1],
},{
  input: [1, 1],
  output: [0],
}];

var network = new Network(2, 1);
var MAX_ITER = 80000;

for (var iter = 0; iter < MAX_ITER; iter++) {
  var i = Math.floor(Math.random() * data.length);
  var input = data[i].input;
  var output = data[i].output;

  var hs = network.forward(input);

  var outputError = math.arraySubtract(hs, output);
  network.backwardError(outputError);
  network.updateWeights();

  if (iter % 250 === 0) console.log('accuracy at iter %s: %s', iter, accuracy());
}

for (var i = 0; i < data.length; i++) {
  var input = data[i].input;
  var output = data[i].output;

  var hs = network.forward(input);

  // Because we know there's only one output.
  var h = hs[0];

  console.log('XOR %s -> %s', data[i].input, h);
}

function accuracy() {
  var correct = 0;
  for (var i = 0; i < data.length; i++) {
    var input = data[i].input;
    var output = data[i].output;

    var hs = network.forward(input);

    // Because we know there's only one output.
    var h = hs[0];

    h = h > 0.5 ? 1 : 0;
    if (h === output[0]) correct++;
  }
  return correct / data.length;
}
