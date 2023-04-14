package moa.classifiers.functions;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import com.github.javacliparser.IntOption;
import com.github.javacliparser.StringOption;
import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.Instances;
import com.yahoo.labs.samoa.instances.InstancesHeader;

import moa.classifiers.AbstractClassifier;
import moa.classifiers.Regressor;
import moa.classifiers.core.driftdetection.ChangeDetector;
import moa.core.Measurement;
import moa.core.Utils;
import no.uib.cipr.matrix.DenseMatrix;
import no.uib.cipr.matrix.DenseVector;
import no.uib.cipr.matrix.Matrix;
import no.uib.cipr.matrix.UpperSPDDenseMatrix;
import no.uib.cipr.matrix.Vector;

public class AdaptiveLinearRegression extends AbstractClassifier implements Regressor{

	private static final long serialVersionUID = 1L;
	
		
	public IntOption numberOfJobsOption = new IntOption("numberOfJobs", 'j',
	        "Total number of concurrent jobs used for processing (-1 = as much as possible, 0 = do not use multithreading)", -1, -1, Integer.MAX_VALUE);
	  
	public StringOption SlidingWindowSizeOption = new StringOption("SlidingWindowSize", 'w',
            "", "100,250,500,1000,1500,2000");
	
    	
	protected static final int SINGLE_THREAD = 0;
	
	protected int instancesSeen;
	protected ChangeDetector driftDetectionMethod;
	protected boolean useDriftDetector;
    protected double examplesSeen;
    protected double[] m_Means;
    protected double[] m_StdDevs;
    protected double m_ClassMean;
    protected double m_ClassStdDev;

    List<Models> listModels; 
    
    protected Models bestModel;
    
    protected boolean removeModel;
    
    protected int GP;
    protected int Janela_Treinamento;
    
   protected int[] arrayWindow;
    
    protected boolean firstTime;
    
    protected double examplesSeenAtLastSplitEvaluation = 0;
    
    protected Instances currentChunk;
    protected Instances chunkTrainTest;
    
    protected Instance lastInst;
    
    protected double classLast;
    
    
    transient private ExecutorService executor;
    
    
	@Override
	public boolean isRandomizable() {
		// TODO Auto-generated method stub
		return true;
	}

	@Override
	public double[] getVotesForInstance(Instance inst) {
		if (this.listModels == null)
			Init(inst);
		
		double prediction = predictionBestModel(inst);
		
        return new double[] {prediction};
	}
	
	private double predictionBestModel(Instance inst) {
		double prediction = 0;
		if (bestModel == null) 
			return classLast;
		
		prediction = bestModel.prediction(inst);
		return prediction;
	}

	@Override
	public void resetLearningImpl() {
		reset();
		instancesSeen = 0;
		
		// Multi-threading
        int numberOfJobs;
        if(this.numberOfJobsOption.getValue() == -1) 
            numberOfJobs = Runtime.getRuntime().availableProcessors();
        else 
            numberOfJobs = this.numberOfJobsOption.getValue();
        // SINGLE_THREAD and requesting for only 1 thread are equivalent. 
        // this.executor will be null and not used...
        if(numberOfJobs != AdaptiveLinearRegression.SINGLE_THREAD && numberOfJobs != 1)
            this.executor = Executors.newFixedThreadPool(numberOfJobs);
	}

	@Override
	public void trainOnInstanceImpl(Instance inst) {
		++this.instancesSeen;
		if (this.listModels == null)
			Init(inst);
		
        examplesSeen += inst.weight();
        int n = inst.numAttributes()-1; // Ignore class label ( -1 )
        GP = inst.numAttributes() * 5;
       
        
        int indexBest = -1;
        double mse = Double.MAX_VALUE;
        for(Models model:listModels) {
        	
        	if (model.getSelectedAttributes() != null) {

        		if (!model.selected )
        			continue;

        		model.setmse(inst);
        		double currentMse = model.getmse();
        		if (currentMse< mse) {
        			indexBest = model.index;
        			mse = currentMse;
        		}
        	}
        }
				
				
		boolean train = true;
        if (indexBest >= 0) {
        	double mean =  listModels.get(indexBest).mean_mse;
        	
        	if (listModels.get(indexBest).valor_mse <=  mean)
        		train = false;
        }
        
       
       
        for(Models model:listModels) {
        	model.addInstance(inst);
        	
        	if (instancesSeen-1 == model.train_window) {
        		for(Models model2:listModels) {
        			model2.resetMse();
        		}
        	}
        	
        }
       
        
        if (instancesSeen >= GP && train) {
       
			Collection<TrainingRunnable> trainers = new ArrayList<TrainingRunnable>();   
			for (int i = 0; i < arrayWindow.length; i++) {
				if (!listModels.get(i).selected )
					continue;
				
				if (instancesSeen >= arrayWindow[i]) {
					
					if(this.executor != null) {
	                    TrainingRunnable trainer = new 
	                    		TrainingRunnable(listModels.get(i), inst, n);
	                    trainers.add(trainer);
	                }
					else {
						try {
							listModels.get(i).updateModel(n, inst);
						} catch (Exception e) {
							e.printStackTrace();
						}
	                }
					
				}
			}
			
			if(this.executor != null) {
				try {
	            	this.executor.invokeAll(trainers);
	            } catch (InterruptedException ex) {
	            	Thread.currentThread().interrupt();
	                throw new RuntimeException("Could not call invokeAll() on training threads.");
	            } 
	        }
			
		}
		
				
		if (indexBest >= 0) {
			bestModel = listModels.get(indexBest);
			bestModel.setCounter();
			
			if (instancesSeen == arrayWindow[arrayWindow.length-1] + 500) {
						
				double best = Double.MIN_VALUE;
				int idxbest = -1;
				for(Models model:listModels) {
					if (model.selected ) {
						if (model.counter > best ) {
							best = model.counter;
							idxbest = model.index;
						}
						
					}
				}
				
				for(Models model:listModels) {
					if (idxbest != model.index)
						model.setSelected(false);
				}
				
			}
			
		}
		else
			bestModel = null;
		
		this.classLast = inst.classValue(); 
		
		
	}
	
	
	
	protected double regressionPrediction(Instance transformedInstance,
		    boolean[] selectedAttributes, double[] coefficients) throws Exception {

		    double result = 0;
		    int column = 0;
		    for (int j = 0; j < selectedAttributes.length; j++) {
		      if (selectedAttributes[j]) {
		        result += coefficients[column] * transformedInstance.value(j);
		        column++;
		      }
		    }
		    result += coefficients[column];

		    return result;
	}
	
	private void Init(Instance inst) {
		bestModel = null;
		Janela_Treinamento = 0;
		GP = 0;
		firstTime = true;
		
		this.currentChunk = new Instances(this.getModelContext());
		
		listModels = new ArrayList<Models>();
		
		int n = inst.numAttributes()-1;
		
		String rawOptions = SlidingWindowSizeOption.getValue();
	    String[] options = rawOptions.split(",");
	    arrayWindow = new int[options.length+1];

	    arrayWindow[0] = 1;
	    int index = 1;
	    for (String string : options) {
		    arrayWindow[index] = Integer.valueOf(string);
		    index++;
		}
	    
		for (int i = 0; i < arrayWindow.length; i++) {
			Models model = new Models();
			model.index = i;
			model._WINDOW_ERROR = n*5;
			model.train_window= arrayWindow[i]; 
			model.setSelected(true);
			model.createChunk(this.getModelContext());
			model.resetMse();
			listModels.add(model);
		}
		
	}



	private void reset() {
		examplesSeen = 0;
	}

	@Override
	protected Measurement[] getModelMeasurementsImpl() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public void getModelDescription(StringBuilder out, int indent) {
		// TODO Auto-generated method stub
		
	}
	
	
	public static class Models {
		protected int index;
		protected int train_window;
		protected int counter;
		protected boolean selected;
		protected double examplesSeen;
		private boolean[] selectedAttributes;
		protected double[] coeffs;
		
		protected double[] mse;
		protected int idx_mse;
		
		protected double current_mse;
		protected double last_mse;
		
		protected double mean_mse; 
		protected double stdDevs_mse; 
		protected double valor_mse; 
		
		protected int n;
		protected int k;
		protected double yLast;
		
		protected Instances currentChunk;
	    protected double m_ClassMean;
	    protected double m_ClassStdDev;
	    protected double[] m_Means;
	    protected double[] m_StdDevs;
	    
	    protected int _WINDOW_ERROR;
	    
	    protected double[] VectorErrorSquares;
	    protected int idxVector;
	    
	    
		public Models() {
			examplesSeen = 0;
			selected = false;
			k=0;
			yLast = 0;
			idx_mse= 0;
		}
		
		public void createChunk(InstancesHeader modelContext) {
			this.currentChunk = new Instances(modelContext);
		}

		public void addInstance(Instance inst) {
			this.currentChunk.add(inst);
			if (currentChunk.size() > train_window) 
				this.currentChunk.delete(0);
			
		}
		
		public void resetMse() {
			mse = new double[_WINDOW_ERROR+1];
			counter = 0;
		}

		public void setCounter() {
			counter++;
		}
		
		public void setmse(Instance inst) {
			double prediction =
			        regressionPrediction(inst, selectedAttributes, coeffs);
			double error = prediction - inst.classValue();
			
			valor_mse = error*error;
					
			mse[mse.length-1] -= mse[idx_mse];
			mse[idx_mse] = error*error;
			mse[mse.length-1] += mse[idx_mse];
			
			idx_mse++;
			if (idx_mse == mse.length-1) {
				idx_mse = 0;
			}
		}
		
		
		private double getmse() {
			last_mse =  current_mse;
			double value = mse[mse.length-1];
			
			mean_mse = value / mse.length;
			current_mse = value;
			return value; 
			
		}

		public void standardDeviation() {
			double standardDeviation = 0.0;
			double res = 0.0;
			for (int i = 0; i < mse.length; i++) {
		           
	            standardDeviation
	                = standardDeviation + Math.pow((mse[i] - mean_mse), 2);
	           
	        }
	       
			double sq = standardDeviation / mse.length;
	        res = Math.sqrt(sq);
	        stdDevs_mse = res;
		}
		
		
		private double[] doRegression(Instances localChunk, boolean[] selectedAttributes) {
			int numAttributes = 0;
			for (boolean selectedAttribute : selectedAttributes) {
				if (selectedAttribute) {
					numAttributes++;
				}
			}

			int m_ClassIndex = localChunk.classIndex();
			// Check whether there are still attributes left
			Matrix independentTransposed = null;
			//Matrix independent = null;
			Vector dependent = null;
			if (numAttributes > 0) {
				independentTransposed = new DenseMatrix(numAttributes, localChunk.numInstances());


				dependent = new DenseVector(localChunk.numInstances());
				for (int i = 0; i < localChunk.numInstances(); i++) {
					Instance inst = localChunk.instance(i);
					double sqrt_weight = Math.sqrt(inst.weight());
					int index = 0;
					for (int j = 0; j < localChunk.numAttributes(); j++) {
						if (j == m_ClassIndex) {
							dependent.set(i, inst.classValue() * sqrt_weight);
						} else {
							if (selectedAttributes[j]) {
								double value = inst.value(j) - m_Means[j];

								// scale the input
								value /= m_StdDevs[j];
								independentTransposed.set(index, i, value * sqrt_weight);
								index++;
							}
						}
					}
				}
			}
			
			double[] coefficients = new double[numAttributes + 1];
			if (numAttributes > 0) {

				// Use Cholesky based on covariance matrix
				Vector aTy = independentTransposed.mult(dependent, new DenseVector(numAttributes));
				Matrix aTa = new UpperSPDDenseMatrix(numAttributes).rank1(independentTransposed);
				independentTransposed = null;
				dependent = null;

				double ridge = getRidge();
				for (int i = 0; i < numAttributes; i++) {
					aTa.add(i, i, ridge);
				}
				Vector coeffsWithoutIntercept = aTa.solve(aTy, new DenseVector(numAttributes));
				System.arraycopy(((DenseVector) coeffsWithoutIntercept).getData(), 0, coefficients, 0, numAttributes);
			}
			coefficients[numAttributes] = m_ClassMean;

			// Convert coefficients into original scale
		    int column = 0;
		    for (int i = 0; i < localChunk.numAttributes(); i++) {
		      if ((i != localChunk.classIndex()) && (selectedAttributes[i])) {

		    	coefficients[column] /= m_StdDevs[i];

		        // We have centred the input
		        coefficients[coefficients.length - 1] -=
		          coefficients[column] * m_Means[i];
		        column++;
		      }
		    }

		    return coefficients;
		}
		
		private double getRidge() {
			 return 1.0e-8;
		}

		private double stdDevs(Instances localChunk, int j) {
			return Math.sqrt(variance(localChunk, j));
		}

		private double variance(Instances localChunk, int attIndex) {
			double mean = 0;
		    double var = Double.NaN;
		    double sumWeights = 0;
		    for(int i = 0; i < localChunk.size(); i++){
		      if (!localChunk.instance(i).isMissing(attIndex)) {
		        double weight = localChunk.instance(i).weight();
		        double value = localChunk.instance(i).value(attIndex);

		        if (Double.isNaN(var)) {
		          // For the first value the mean can suffer from loss of precision
		          // so we treat it separately and make sure the calculation stays accurate
		          mean = value;
		          sumWeights = weight;
		          var = 0;
		          continue;
		        }

		        double delta = weight*(value - mean);
		        sumWeights += weight;
		        mean += delta/sumWeights;
		        var += delta*(value - mean);
		      }
		    }

		    if (sumWeights <= 1) {
		      return Double.NaN;
		    }

		    var /= sumWeights - 1;

		    // We don't like negative variance
		    if (var < 0) {
		      return 0;
		    } else {
		      return var;
		    }
		}

		private double meanOrMode(Instances localChunk, int attIndex) {
			
			double result, found;
		    int[] counts;

		    if (localChunk.instance(0).attribute(attIndex).isNumeric()) {
		      result = found = 0;
		      for(int i = 0; i < localChunk.size(); i++){
		        if (!localChunk.instance(i).isMissing(attIndex)) {
		          found += localChunk.instance(i).weight();
		          result += localChunk.instance(i).weight() * localChunk.instance(i).value(attIndex);
		        }
		      }
		      if (found <= 0) {
		        return 0;
		      } else {
		        return result / found;
		      }
		    } else if (localChunk.instance(0).attribute(attIndex).isNominal()) {
		      counts = new int[localChunk.instance(0).attribute(attIndex).numValues()];
		      for (int j = 0; j < localChunk.size(); j++) {
		    	  if (!localChunk.instance(j).isMissing(attIndex)) {
		          counts[(int) localChunk.instance(j).value(attIndex)] += localChunk.instance(j).weight();
		        }
		      }
		      return Utils.maxIndex(counts);
		    } else {
		      return 0;
		    }
		    
		}
		
		public double[] getCoeffs() {
			return coeffs;
		}

		public void setCoeffs(double[] coeffs) {
			this.coeffs = coeffs;
		}

		public boolean isSelected() {
			return selected;
		}

		public void setSelected(boolean selected) {
			this.selected = selected;
		}
		
		public boolean[] getSelectedAttributes() {
			return selectedAttributes;
		}

		public void setSelectedAttributes(boolean[] selectedAttributes) {
			this.selectedAttributes = selectedAttributes;
		}
				
		public void update(Instance inst) {
			if (k == 0) 
				init();
	    }
		
		private void init() {
			k = 0;
			for (int i = 0; i < selectedAttributes.length; i++) {
				if (selectedAttributes[i])
					k++;
			}
		}

		protected double prediction(Instance inst) {
			return regressionPrediction(inst, selectedAttributes, coeffs);
	    }
		
		protected double regressionPrediction(Instance transformedInstance,
			    boolean[] selectedAttributes, double[] coefficients)  {
			
			double result = 0;
			int column = 0;
			for (int j = 0; j < selectedAttributes.length; j++) {
				if (selectedAttributes[j]) {
					result += coefficients[column] * transformedInstance.value(j);
					column++;
				}
			}
			result += coefficients[column];

			return result;
		}
		
		
		private double calculateSE(boolean[] selectedAttributes, double[] coefficients, int limite) {
			double mse = 0;
			for(int i = currentChunk.size()-1; i >= limite; i--){
			      double prediction =
			        regressionPrediction(currentChunk.instance(i), selectedAttributes, coefficients);
			      double error = prediction - currentChunk.instance(i).classValue();
			      mse += error * error;
			 }

		    return mse;
		}
		
		public double getAkaike() {
			
			if (currentChunk == null)
				return 0;
			
			int limite = 0;
			if (currentChunk.numInstances() > _WINDOW_ERROR)
				limite = currentChunk.numInstances()-_WINDOW_ERROR;
			
			double currentMSE = calculateSE(selectedAttributes, coeffs, limite);
			
			int numInstances;
			if (currentChunk.numInstances() > _WINDOW_ERROR)
				numInstances = _WINDOW_ERROR;
			else
				numInstances = currentChunk.numInstances();
			
			double aic = currentMSE * (numInstances - k) + 2* k;
						 
			return aic;
			
		}
		
		
		
		public double getR2(double targetMean, double ySquares, double examplesSeen2) {
			int limite = 0;
			if (currentChunk.numInstances() > _WINDOW_ERROR)
				limite = currentChunk.numInstances()-_WINDOW_ERROR;
			
			
			double examplesSeen = (examplesSeen2 > _WINDOW_ERROR ? _WINDOW_ERROR : examplesSeen2);
			
			double SST = ySquares - (examplesSeen*(targetMean*targetMean));
			
			double errorSquares = calculateSE(selectedAttributes, coeffs, limite);
			double SSR = SST - errorSquares;
			return SSR/SST;
		}

		public double getR2_Original() {
			double examplesSeen = (currentChunk.numInstances() > _WINDOW_ERROR ? _WINDOW_ERROR : currentChunk.numInstances());
			
			int limite = 0;
			if (currentChunk.numInstances() > _WINDOW_ERROR)
				limite = currentChunk.numInstances()-_WINDOW_ERROR;
			
			double Ymean = getMean(limite);
			double SST = getYSquares(limite) - (examplesSeen*(Ymean*Ymean));
			
			double currentMSE = calculateSE(selectedAttributes, coeffs, limite);
			double SSR = SST - currentMSE;
			return SSR/SST;
		}

		
		private double getYSquares(int limite) {
			double ySquares = 0;
			
			for(int i = currentChunk.size()-1; i >= limite; i--){
				ySquares += currentChunk.instance(i).weight() *(currentChunk.instance(i).classValue()*currentChunk.instance(i).classValue());
			 }
			
			return ySquares;
		}


		private double getMean(int limite) {
			
			double y = 0;
			int count = 0;
			for(int i = currentChunk.size()-1; i >= limite; i--){
			      y += currentChunk.instance(i).classValue();
			      count++;
			 }
			return y/count;
		}

		
		private void updateModel(int n, Instance inst) {
			examplesSeen++;
			
			Instances localChunk = this.currentChunk;
			
			int numInstances = localChunk.size();
			
			double [] m_Coefficients = null;
			boolean [] m_SelectedAttributes = new boolean[n];
			
		    m_ClassStdDev = stdDevs(localChunk, localChunk.classIndex());
		    m_ClassMean  = meanOrMode(localChunk, localChunk.classIndex());
			
		    if (numInstances == 1) { // intercept only
			    for (int j = 0; j < m_SelectedAttributes.length; j++) {
			    	m_SelectedAttributes[j] = false;
			    }
			    
			    m_Coefficients = doRegression(localChunk, m_SelectedAttributes);
		    }
		    else {
		    	
		    	m_Means = new double[m_SelectedAttributes.length];
				m_StdDevs = new double[m_SelectedAttributes.length];
			    for (int j = 0; j < m_SelectedAttributes.length; j++) {
			    	m_SelectedAttributes[j] = true; // Turn attributes on for a start
			    	m_Means[j] = meanOrMode(localChunk, j);
			    	m_StdDevs[j] = stdDevs(localChunk, j);
			    	if (m_StdDevs[j] == 0) {
			    		m_SelectedAttributes[j] = false;
			    	}
			    }
			    
		    	do {
		    		m_Coefficients = doRegression(localChunk, m_SelectedAttributes);
		    	} while (deselectColinearAttributes(m_SelectedAttributes, m_Coefficients));
		    	
			  
		    	
			    int numAttributes = 1;
			    for (boolean m_SelectedAttribute : m_SelectedAttributes) {
			      if (m_SelectedAttribute) {
			        numAttributes++;
			      }
			    }

			    double fullMSE = 0;
			    try {
					fullMSE = calculateSE(localChunk, m_SelectedAttributes, m_Coefficients);
				} catch (Exception e) {
					e.printStackTrace();
				}
			    double akaike = (numInstances - numAttributes) + 2 * numAttributes;

			    boolean improved;
			    int currentNumAttributes = numAttributes;
			    
			    int count = 0;
			  //M5
			    do {
			    	count++;
			    	
			    	improved = false;
			        currentNumAttributes--;

			        // Find attribute with smallest SC
			        double minSC = 0;
			        int minAttr = -1, coeff = 0;
			        for (int i = 0; i < m_SelectedAttributes.length; i++) {
			        	if (m_SelectedAttributes[i]) {
			        		double SC =
			        				Math.abs(m_Coefficients[coeff] * m_StdDevs[i] / m_ClassStdDev);
			        		if ((coeff == 0) || (SC < minSC)) {
			        			minSC = SC;
			        			minAttr = i;
			        		}
			        		coeff++;
			        	}
			        }

			        // See whether removing it improves the Akaike score
			        if (minAttr >= 0) {
			        	m_SelectedAttributes[minAttr] = false;
			        	double[] currentCoeffs = doRegression(localChunk, m_SelectedAttributes);
			        	double currentMSE = 0;
			        	try {
			        		currentMSE = calculateSE(localChunk, m_SelectedAttributes, currentCoeffs);
			        	} catch (Exception e) {
			        		e.printStackTrace();
			        	}
			        	double currentAkaike =
			        			currentMSE / fullMSE * (numInstances - numAttributes) + 2
			        			* currentNumAttributes;

			        	// If it is better than the current best
			        	if (currentAkaike < akaike) {
			        		improved = true;
			        		akaike = currentAkaike;
			        		m_Coefficients = currentCoeffs;
			        	} else {
			        		m_SelectedAttributes[minAttr] = true;
			        	}
			        }
			        
			      } while (improved);
		    }
			
		    boolean update = false;
		   
		    
		    double mseOld = Double.MAX_VALUE;
		    double prediction;
		    double error;
		    
		    if (selectedAttributes != null && coeffs != null) {
			    prediction =
		        regressionPrediction(inst, selectedAttributes, coeffs);
			    error = prediction - inst.classValue();
		      	mseOld = error * error;
		    }
		    
	      	prediction =
	    	        regressionPrediction(inst, m_SelectedAttributes,
	    	        		m_Coefficients);
	      	error = prediction - inst.classValue();
	    	double mseNew = error * error;
		    
	    	if (mseNew < mseOld)
	    		update = true;
	    	
	    	if (update) {
	    		selectedAttributes = m_SelectedAttributes ;
	    		coeffs = m_Coefficients;
		    }
	    	
		}
		
		public static double calculateStandardDeviation(double[] array, double mean) {

		    // get the mean of array
		    int length = array.length;

		    // calculate the standard deviation
		    double standardDeviation = 0.0;
		    for (double num : array) {
		        standardDeviation += Math.pow(num - mean, 2);
		    }

		    return Math.sqrt(standardDeviation / length);
		}
		
	    
		protected double calculateSE(Instances localChunk, boolean[] selectedAttributes,
				double[] coefficients) throws Exception {

			double mse = 0;
			for (int i = 0; i < localChunk.numInstances(); i++) {
				double prediction =
						regressionPrediction(localChunk.instance(i), selectedAttributes,
								coefficients);
				double error = prediction - localChunk.instance(i).classValue();
				mse += error * error;
			}
			return mse;
		}
		
		protected boolean deselectColinearAttributes(boolean[] selectedAttributes,
				double[] coefficients) {

			double maxSC = 1.5;
			int maxAttr = -1, coeff = 0;
			for (int i = 0; i < selectedAttributes.length; i++) {
				if (selectedAttributes[i]) {
					double SC =
							Math.abs(coefficients[coeff] * m_StdDevs[i] / m_ClassStdDev);
					if (SC > maxSC) {
						maxSC = SC;
						maxAttr = i;
					}
					coeff++;
				}
			}
			if (maxAttr >= 0) {
				selectedAttributes[maxAttr] = false;
				return true;
			}
			return false;
		}


	}
	
	/***
     * Inner class to assist with the multi-thread execution. 
     */
    protected class TrainingRunnable implements Runnable, Callable<Integer> {
        final private Models model;
        final private Instance inst;
        final private int n;

        public TrainingRunnable(Models model, Instance inst, int n) {
            this.model = model;
            this.inst = inst;
            this.n = n;
        }

        @Override
        public void run() {
        	model.updateModel(n, inst);
        }

        @Override
        public Integer call() {
            run();
            return 0;
        }
    }
	
	
}

