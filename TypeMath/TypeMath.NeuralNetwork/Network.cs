using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.Distributions;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.Random;

namespace TypeMath.NeuralNetwork
{
    public class Network
    {
        #region Construction
        public Network(int inputNeurons, int outputNeurons, int hiddenNeurons)
        {
            this.weights = new Matrix<double>[2];
            this.activations = new Vector<double>[3];

            this.InitWeights(inputNeurons, hiddenNeurons, 0);
            this.InitWeights(hiddenNeurons, outputNeurons, 1);
        }
        #endregion

        #region Public methods
        public void Train(IEnumerable<double> dataEntry, double learningConst)
        {
            var attributes = dataEntry.Take(dataEntry.Count() - 1);
            var expectedClass = (int)dataEntry.Last();

            this.Train(attributes, expectedClass, learningConst);
        }

        public void Train(IEnumerable<double> dataEntry, int expectedClass, double learningConst)
        {
            var attrVector = Vector<double>.Build.DenseOfEnumerable(dataEntry);
            this.Activate(attrVector);

            var expectedOutput = this.ExpectedOutput(expectedClass, this.activations[2].Count);

            var outputErrors = this.OutputErrors(expectedOutput);
            var hiddenLayerErrors = this.HiddenLayerErrors(this.activations[1], outputErrors);

            this.UpdateWeights(this.weights[1], activations[1], outputErrors, learningConst);
            this.UpdateWeights(this.weights[0], activations[0], hiddenLayerErrors, learningConst);
        }        

        public void Train(IList<IEnumerable<double>> data, int iterations, double learningConst)
        {
            for (int i = 0; i < iterations; i++)
            {
                foreach (var entry in data)
                {
                    this.Train(entry, learningConst);
                }
            }
        }

        public void Train(IList<IEnumerable<double>> data, IList<int> labels, int iterations, double learningConst)
        {
            for (int i = 0; i < iterations; i++)
            {
                for (int j = 0; j < data.Count; j++)
                {
                    this.Train(data[j], labels[j], learningConst);
                }
            }
        }

        public Vector<double> GetOutput()
        {
            return this.activations.Last();
        }

        public IList<double[]> ComputeData(IList<IEnumerable<double>> data)
        {
            var results = new List<double[]>();
            foreach (var item in data)
            {
                var vector = Vector<double>.Build.DenseOfEnumerable(item);
                this.Activate(vector);
                var result = this.GetOutput();
                results.Add(result.ToArray());
            }

            return results;
        }
        #endregion

        #region Private methods
        private void InitWeights(int firstLayerNeurons, int secondLayerNeurons, int index)
        {
            // Create matrix filled with random numbers from -0.005 to 0.005
            var weights = Matrix<double>.Build.Random(secondLayerNeurons,
                                                      firstLayerNeurons,
                                                      new ContinuousUniform(-0.05, 0.05, new MersenneTwister(133)));

            this.weights[index] = weights;
        }

        private void Activate(Vector<double> input)
        {
            this.activations[0] = input;

            for (int i = 0; i < this.weights.Length; i++)
            {
                var activations = this.weights[i] * this.activations[i];
                this.activations[i + 1] = activations.Map(x => this.Logistic(x));
            }
        }

        private double Logistic(double sum)
        {
            return 1 / (1 + Math.Pow(Math.E, (-sum)));
        }

        private Vector<double> ExpectedOutput(int expectedClass, int outputNeurons)
        {
            var output = Vector<double>.Build.Sparse(outputNeurons);
            output[expectedClass] = 1.0;
            return output;
        }

        private Vector<double> OutputErrors(Vector<double> expected)
        {
            var output = this.GetOutput();
            return output.MapIndexed((i, x) => x * (1 - x) * (expected[i] - x));
        }

        private Vector<double> HiddenLayerErrors(Vector<double> activations, Vector<double> outputErrors)
        {
            var errors = this.weights[1].Transpose() * outputErrors.ToColumnMatrix();
            var vector = errors.EnumerateColumns().First();
            vector.MapIndexedInplace((i, x) => activations[i] * (1 - activations[i]) * x);
            return vector;
        }

        private void UpdateWeights(Matrix<double> weights, Vector<double> activations, Vector<double> errors, double learningConst)
        {
            weights.MapIndexedInplace(
                (r, c, x) => x + learningConst * errors[r] * activations[c]);
        }
        #endregion

        #region Private fields and constants
        private Matrix<double>[] weights;
        private Vector<double>[] activations;
        #endregion
    }
}
