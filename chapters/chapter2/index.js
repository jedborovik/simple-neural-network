var Neuron = require('./neuron.js');
// var Neuron = require('./neuron.es6');

var data = [{
  input: [0, 0],
  output: 0,
},{
  input: [0, 1],
  output: 0,
},{
  input: [1, 0],
  output: 0,
},{
  input: [1, 1],
  output: 1,
}];

var neuron = new Neuron();

for (var iter = 0; iter < 750; iter++) {
  var i = Math.floor(Math.random() * data.length);
  var input = data[i].input;
  var output = data[i].output;
  input = [1].concat(input);

  var h = neuron.forward(input);

  var error = h - output;
  neuron.updateWeights(error);

  if (iter % 25 === 0) console.log('accuracy at iter %s: %s', iter, accuracy());
}

// Log our final weights.
console.log(neuron.weights);

for (var i = 0; i < data.length; i++) {
  var input = data[i].input;
  input = [1].concat(input);
  var h = neuron.forward(input);
  console.log('%s -> %s', data[i].input, h);
}

function accuracy() {
  var correct = 0;
  for (var i = 0; i < data.length; i++) {
    var input = data[i].input;
    input = [1].concat(input);
    var output = data[i].output;

    var h = neuron.forward(input) > 0.5 ? 1 : 0;
    if (h === output) correct++;
  }
  return correct / data.length;
}
