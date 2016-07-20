package opt.test;

import java.util.Arrays;

import dist.DiscreteDependencyTree;
import dist.DiscreteUniformDistribution;
import dist.Distribution;

import opt.DiscreteChangeOneNeighbor;
import opt.EvaluationFunction;
import opt.GenericHillClimbingProblem;
import opt.HillClimbingProblem;
import opt.NeighborFunction;
import opt.RandomizedHillClimbing;
import opt.SimulatedAnnealing;
import opt.example.*;
import opt.ga.CrossoverFunction;
import opt.ga.DiscreteChangeOneMutation;
import opt.ga.SingleCrossOver;
import opt.ga.GenericGeneticAlgorithmProblem;
import opt.ga.GeneticAlgorithmProblem;
import opt.ga.MutationFunction;
import opt.ga.StandardGeneticAlgorithm;
import opt.prob.GenericProbabilisticOptimizationProblem;
import opt.prob.MIMIC;
import opt.prob.ProbabilisticOptimizationProblem;
import shared.FixedIterationTrainer;

/**
 * Copied from ContinuousPeaksTest
 * @version 1.0
 */
public class FourPeaksTest {
    /** The n value */
    private static final int N = 150;
    /** The t value */
    private static final int T = N / 10;
    
    public static void main(String[] args) {
        int[] ranges = new int[N];
        Arrays.fill(ranges, 2);
        EvaluationFunction ef = new FourPeaksEvaluationFunction(T);
        Distribution odd = new DiscreteUniformDistribution(ranges);
        NeighborFunction nf = new DiscreteChangeOneNeighbor(ranges);
        MutationFunction mf = new DiscreteChangeOneMutation(ranges);
        CrossoverFunction cf = new SingleCrossOver();
        Distribution df = new DiscreteDependencyTree(.1, ranges); 
        HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, odd, nf);
        GeneticAlgorithmProblem gap = new GenericGeneticAlgorithmProblem(ef, odd, mf, cf);
        ProbabilisticOptimizationProblem pop = new GenericProbabilisticOptimizationProblem(ef, odd, df);
        
        RandomizedHillClimbing rhc = new RandomizedHillClimbing(hcp);      
        FixedIterationTrainer fit = new FixedIterationTrainer(rhc, 200000);
        fit.train();
        System.out.println("RHC: " + ef.value(rhc.getOptimal()));
        
        SimulatedAnnealing sa = new SimulatedAnnealing(1E11, .95, hcp);
        fit = new FixedIterationTrainer(sa, 200000);
        fit.train();
        System.out.println("SA: " + ef.value(sa.getOptimal()));
        
        StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm(200, 100, 10, gap);
        fit = new FixedIterationTrainer(ga, 1000);
        fit.train();
        System.out.println("GA: " + ef.value(ga.getOptimal()));
        
        
        double start = System.nanoTime();
        MIMIC mimic = new MIMIC(1000, 30, pop);
        fit = new FixedIterationTrainer(mimic, 1500);
        fit.train();
        double end = System.nanoTime() - start;
        
        end /= Math.pow(10,9);
        
        System.out.println("MIMIC: " + ef.value(mimic.getOptimal()));
        
        System.out.println("\n\n");
        System.out.println(end);
        
        /* Try out different parameters for MIMIC */
        
        // first test out different number of samples
        int[] numSamplesArray = {200,400,600,800,1000}; 
            int i;
        int numSamples;
        
        for(i = 0; i < numSamplesArray.length; i++)
        {   
            numSamples = numSamplesArray[i];
            start = System.nanoTime();
            mimic = new MIMIC(numSamples, 30, pop);
            fit = new FixedIterationTrainer(mimic, 2000);
            fit.train();
            end = System.nanoTime() - start;

            end /= Math.pow(10,9);
            
            
            System.out.println("Number of Samples: ");
            System.out.println(numSamplesArray[i]);

            System.out.println("MIMIC: " + ef.value(mimic.getOptimal()));

            System.out.println("\n\n");
            System.out.println(end);
        }
        
        /* Test out varying numbers of samples to kept from each iterative
           distribution */
        
        int[] keepSamplesArray = {10,30,50,100,200}; 
        int keepSamples;
        
        for(i = 0; i < keepSamplesArray.length; i++)
        {   
            keepSamples = keepSamplesArray[i];
            start = System.nanoTime();
            mimic = new MIMIC(1000, keepSamples, pop);
            fit = new FixedIterationTrainer(mimic, 1500);
            fit.train();
            end = System.nanoTime() - start;

            end /= Math.pow(10,9);
            
            
            System.out.println("Number of Kept Samples: ");
            System.out.println(keepSamplesArray[i]);

            System.out.println("MIMIC: " + ef.value(mimic.getOptimal()));

            System.out.println("\n\n");
            System.out.println(end);
        }
        
        
        
    }
}
