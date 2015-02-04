using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using TypeMath.NeuralNetwork;

namespace TypeMath.Cmd
{
    class CrohmeData
    {
        public async void Learn()
        {
            var startTime = DateTime.Now;

            var trainDataFileName = "../../../data/crohme";
            var trainDataLabelsFileName = "../../../data/crohme";
            StreamReader data = new StreamReader(new MemoryStream());
            var imageTask = CrohmeData.ReadImageDataAsync(trainDataFileName, data);

            StreamReader labels = new StreamReader(new MemoryStream());
            var labelsTask = CrohmeData.ReadLabelsDataAsync(trainDataLabelsFileName, labels);

            var hiddenLayerNeurons = 300;
            var iterations = 2;
            var learningConst = 0.5;
            var net = new Network(784, 10, hiddenLayerNeurons);
            net.Train(data, labels, iterations, learningConst, ' ');

            await imageTask;
            await labelsTask;

            var testDataFileName = "../../../data/crohme";
            var testResultsFileName = "../../../data/crohme";

            var testData = CrohmeData.ReadImageData(testDataFileName);
            var testDataResults = CrohmeData.ReadLabelsData(testResultsFileName);

            var networkResults = net.Classify(testData);

            var endTime = DateTime.Now;

            double successfullTests = 0;

            for (int i = 0; i < testDataResults.Count; i++)
            {
                double max = -1;
                int maxIndex = -1;
                for (int j = 0; j < networkResults[i].Length; j++)
                {
                    if (networkResults[i][j] > max)
                    {
                        max = networkResults[i][j];
                        maxIndex = j;
                    }
                }
                Console.WriteLine("Expected: {0} Actual: {1} Output {2}", testDataResults[i], maxIndex, max);

                if (testDataResults[i] == maxIndex)
                {
                    successfullTests++;
                }
            }

            var time = endTime - startTime;

            Console.WriteLine("Execution time: {0}m {1}s", time.Minutes, time.Seconds);

            Console.WriteLine("Training samples: {0}", testData.Count);
            Console.WriteLine("Successfull tests: {0}", successfullTests);
            Console.WriteLine("Success rate: {0}", successfullTests / testData.Count);
        }

        private static List<IEnumerable<double>> ReadImageData(string fileName)
        {
            var data = new List<IEnumerable<double>>();

            using (var fs = new FileStream(fileName, FileMode.Open))
            {
                var magicNumber = ReadLine(fs);
                var numberOfImages = ReadLine(fs);
                var numberOfRows = ReadLine(fs);
                var numberOfColumns = ReadLine(fs);

                for (int i = 0; i < numberOfImages; i++)
                {
                    var pixels = new List<double>();
                    var numberOfPixels = numberOfRows * numberOfColumns;

                    for (int j = 0; j < numberOfPixels; j++)
                    {
                        var pixel = fs.ReadByte();
                        var inputData = pixel / 255.0d;
                        pixels.Add(inputData);
                    }

                    data.Add(pixels);
                }
            }
            return data;
        }

        private static async Task ReadImageDataAsync(string fileName, StreamReader data)
        {
            using (var fs = new FileStream(fileName, FileMode.Open))
            {
                var magicNumber = ReadLine(fs);
                var numberOfImages = ReadLine(fs);
                var numberOfRows = ReadLine(fs);
                var numberOfColumns = ReadLine(fs);

                var writer = new StreamWriter(data.BaseStream);

                for (int i = 0; i < numberOfImages; i++)
                {
                    var numberOfPixels = numberOfRows * numberOfColumns;
                    var line = new string[numberOfPixels];
                    for (int j = 0; j < numberOfPixels; j++)
                    {
                        var pixel = fs.ReadByte();
                        var inputData = pixel / 255.0d;
                        line[j] = inputData.ToString();
                    }

                    await writer.WriteLineAsync(String.Join(" ", line));
                }

                data.BaseStream.Seek(0, SeekOrigin.Begin);
            }
        }

        private static List<int> ReadLabelsData(string fileName)
        {
            var labels = new List<int>();

            using (var fs = new FileStream(fileName, FileMode.Open))
            {
                var magicNumber = ReadLine(fs);
                var numberOfItems = ReadLine(fs);

                for (int i = 0; i < numberOfItems; i++)
                {
                    var label = fs.ReadByte();
                    labels.Add(label);
                }
            }

            return labels;
        }

        private static async Task ReadLabelsDataAsync(string fileName, StreamReader data)
        {
            var writer = new StreamWriter(data.BaseStream);

            using (var fs = new FileStream(fileName, FileMode.Open))
            {
                var magicNumber = ReadLine(fs);
                var numberOfItems = ReadLine(fs);

                for (int i = 0; i < numberOfItems; i++)
                {
                    var label = fs.ReadByte();
                    await writer.WriteLineAsync(label.ToString());
                }
            }

            data.BaseStream.Seek(0, SeekOrigin.Begin);
        }

        private static int ReadLine(FileStream fs)
        {
            return (fs.ReadByte() << 24) | (fs.ReadByte() << 16) | (fs.ReadByte() << 8) | fs.ReadByte();
        }
    }
}
