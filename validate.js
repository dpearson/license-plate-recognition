var fs = require("fs");

var prompt = require("sync-prompt").prompt;
var shelljs = require("shelljs");

function levenshteinDistance (s, t) {
        if (s.length === 0) return t.length;
        if (t.length === 0) return s.length;

        return Math.min(
                levenshteinDistance(s.substr(1), t) + 1,
                levenshteinDistance(t.substr(1), s) + 1,
                levenshteinDistance(s.substr(1), t.substr(1)) + (s[0] !== t[0] ? 1 : 0)
        );
}

var srcFiles = fs.readdirSync("../train_data/test_images/");

var distances = [];

var exempt = [5, 13, 19, 20, 21, 22/*Fake plate*/, 23, 26, 27, 28, 29, 31, 37, 38, 39, 40, 41];

for (var i = 0; i < srcFiles.length; i++) {
	if (srcFiles[i].toString() === ".DS_Store") {
		continue;
	}

	var filepath = "../train_data/test_images/" + srcFiles[i];

	var number = parseInt(filepath.replace(/[^0-9]/g, ""));
	var shouldUse = true;
	for (var j = 0; j < exempt.length && exempt[j] <= number; j++) {
		if (exempt[j] == number) {
			shouldUse = false;
		}
	}
	if (!shouldUse) {
		continue;
	}

	var predictedText = shelljs.exec("../bin/recognize " + filepath).output.replace(/\s/g, "");
	if (predictedText.indexOf("error") >= 0) {
		continue;
	}

	var annotation = fs.readFileSync("../train_data/annotations/" + srcFiles[i] + ".txt").toString();
	var lines = annotation.split("\n");
	var actualText = lines[4];

	var distance = levenshteinDistance(actualText, predictedText);
	distances.push(distance);

	console.log(actualText + " -> " + predictedText + ", distance = " + distance);
}

var sum = 0;
for (var i = 0; i < distances.length; i++) {
	sum += distances[i];
}
var mean = sum / distances.length;

var variance = 0;
for (var i = 0; i < distances.length; i++) {
	variance += Math.pow(distances[i] - mean, 2);
}
variance *= 1 / (distances.length - 1);

distances = distances.sort(function (a, b) { return a - b; });
var median = distances[Math.floor(distances.length / 2)];

var stdDev = Math.sqrt(variance);

console.log("\n\nmean = " + mean + "\nmedian = " + median + "\nstandard deviation = " + stdDev);