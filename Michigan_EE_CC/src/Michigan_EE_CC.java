import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.Random;
import java.util.stream.Collectors;

public class Michigan_EE_CC {
	public static void main(String[] args) throws FileNotFoundException, IOException {
		String sDS = "segment";
		int nClas = 7;
		int iCV = 10;
		
		int iFold = 3;
		int iResult = 3;
		
		if(args.length > 0) {
			sDS = args[0];
			nClas = Integer.parseInt(args[1]);
			iCV = Integer.parseInt(args[2]);
			iFold = Integer.parseInt(args[3]);
			iResult = Integer.parseInt(args[4]);
		}
		
		//Parameters of the genetic learning of preliminary rule base
		int nFS = 5;
		int N_rule = 30;
		int N_replace = 6;
		double crossProb = 0.9;
		int ni = 100000;
	
		System.out.println("Dataset: " + sDS);
		System.out.println("Cross validation: " + iCV);
		System.out.println("Cross validation fold: " + iFold);
		System.out.println("Number of classes: " + nClas);
		
		System.out.println("************* Parameters of Michigan-style fuzzy GBML *******");
		System.out.println("Number of triangular fuzzy sets: " + nFS);
		System.out.println("Population size: " + N_rule);
		System.out.println("Number or replaced rules: " + N_replace);
		System.out.println("Parent selection: Binary tournament selection");
		System.out.println("Crossover: Uniform crossover");
		System.out.println("Crossover probability: " + crossProb);
		System.out.println("Mutation probability (1/n): n Number of attributes ");
		System.out.println("Number of iterations = " + ni);
		
		System.out.println();
		
		int r = 0;
		
		// Genetic Algorithm - Parameters
		Random rn = new Random();
		
		// Read Train and Test Data
		String docuTrain = System.getProperty("user.dir") + File.separator + "datasets" + File.separator
				+ sDS + "-" + iCV + "-fold" + File.separator + sDS + "-" + iCV + "-" + iFold + "tra.dat";

		String docuTest = System.getProperty("user.dir") + File.separator + "datasets" + File.separator
				+ sDS + "-" + iCV + "-fold" + File.separator + sDS + "-" + iCV + "-" + iFold + "tst.dat";

		//Read train data set
		int nExamTrain = 0;
		String sLine;
		String[] asLine;
		int e;
		int nAttr = 0;
		
		try (BufferedReader br = Files.newBufferedReader(Paths.get(docuTrain))) {
			while ((sLine = br.readLine()) != null) {
				if(sLine.contains("@")) {
					if(sLine.contains("@attribute")) {
						nAttr++;
					}
				} else {
					if(!sLine.trim().isEmpty()) {
						nExamTrain++; 
					}
				}
			}
		}
		nAttr = nAttr - 1;
		
		double dataTrain[][] = new double[nExamTrain][nAttr + 1];
		String[] attr_names = new String[nAttr];
		
		try (BufferedReader br = Files.newBufferedReader(Paths.get(docuTrain))) {
			e = 0;
			while ((sLine = br.readLine()) != null) {
				sLine = sLine.trim();
				if(!sLine.contains("@")) {
					asLine = sLine.split(",");
					for(int c = 0; c < asLine.length - 1; c++) {
						dataTrain[e][c] = Double.parseDouble(asLine[c]);
					}
					if(sDS.equals("pima")) {
						if(asLine[asLine.length - 1].equals("tested_positive")) {
							dataTrain[e][asLine.length - 1] = 2.0;
						} else {
							dataTrain[e][asLine.length - 1] = 1.0;
						}
					} else if (sDS.equals("appendicitis")) {
						dataTrain[e][asLine.length - 1] = Double.parseDouble(asLine[asLine.length - 1]) + 1.0;
					} else {
						dataTrain[e][asLine.length - 1] = Double.parseDouble(asLine[asLine.length - 1]);
					}
					e++;
				} else {
					if(sLine.contains("@inputs")) {
						sLine = sLine.replace("@inputs ", "");
						asLine = sLine.split(",");
						for(int a = 0; a < asLine.length; a++) {
							attr_names[a] = (asLine[a].trim());
						}
					}
				}
			}
		} catch (IOException exc) {
			exc.printStackTrace();
		}

		//Read test data set
		int nExamTest = 0;
		
		try (BufferedReader br = Files.newBufferedReader(Paths.get(docuTest))) {
			while ((sLine = br.readLine()) != null) {
				if(!sLine.contains("@")) {
					if(!sLine.trim().isEmpty()) {
						nExamTest++; 
					}
				}
			}
		}
		
		double dataTest[][] = new double[nExamTest][nAttr + 1];
		
		try (BufferedReader br = Files.newBufferedReader(Paths.get(docuTest))) {
			e = 0;
			while ((sLine = br.readLine()) != null) {
				sLine = sLine.trim();
				if(!sLine.contains("@")) {
					asLine = sLine.split(",");
					for(int c = 0; c < asLine.length - 1; c++) {
						dataTest[e][c] = Double.parseDouble(asLine[c]);
					}
					if(sDS.equals("pima")) {
						if(asLine[asLine.length - 1].equals("tested_positive")) {
							dataTest[e][asLine.length - 1] = 2.0;
						} else {
							dataTest[e][asLine.length - 1] = 1.0;
						}
					} else if (sDS.equals("appendicitis")) {
						dataTest[e][asLine.length - 1] = Double.parseDouble(asLine[asLine.length - 1]) + 1.0;
					} else {
						dataTest[e][asLine.length - 1] = Double.parseDouble(asLine[asLine.length - 1]);
					}
					e++;
				}
			}
		} catch (IOException exc) {
			exc.printStackTrace();
		}
		
		//showMatrizDouble(dataTrain);
		////////////////////////////////////////////////////////////////////////////////////

		// Create Uniformly Fuzzy Sets
		double dMax;
		double dMin;
		double dStep;
		double dFuzzySets[][] = new double[nAttr][nFS * 3];
		int r_aux;
		int c_aux;

		for (int c = 0; c < nAttr; c++) {
			dMax = Double.MAX_VALUE * -1;
			dMin = Double.MAX_VALUE;
			for (r = 0; r < dataTrain.length; r++) {
				if(dataTrain[r][c] != 0) {
					if (dataTrain[r][c] > dMax) {
						dMax = dataTrain[r][c];
					}
					if (dataTrain[r][c] < dMin) {
						dMin = dataTrain[r][c];
					}
				}
			}
			r_aux = c;
			if(nFS != 1) {
				dStep = (dMax - dMin) / (nFS - 1);
				dMin = dMin - dStep;
			}
			else {
				dStep = 1.0;
				dMin = 0.0;
			}
			for (c_aux = 0; c_aux < nFS * 3; c_aux = c_aux + 3) {
				dFuzzySets[r_aux][c_aux] = dMin;
				dFuzzySets[r_aux][c_aux + 1] = dMin + dStep;
				dFuzzySets[r_aux][c_aux + 2] = dMin + dStep + dStep;
				dMin = dMin + dStep;
			}
		}

		//showMatrizDouble(dFuzzySets);

		// Create Fuzzy Rules
		int sizeRule = nAttr + 2;
		ArrayList<double[]> popuP = new ArrayList<>();
		ArrayList<double[]> popuN = new ArrayList<>();
		ArrayList<double[]> popuP_Best = new ArrayList<>();

		double H = 2;
		ArrayList<Integer> multiPattern = new ArrayList<>();
		boolean[] arClasiPatt = new boolean[nExamTrain];
		double[] q = new double[nClas];
		double[] q_Best = new double[nClas];

		int iData;
		double dMembDegr, dClass;
		double dMinMembDegr;
		
		ArrayList<Double> arMembDegr = new ArrayList<>();
		ArrayList<Double> arMinMembDegr = new ArrayList<>();
		double[] rule;
		double[] arFitnRule = new double[N_rule];
		ArrayList<Integer> worst_rules = new ArrayList<>();
		ArrayList<double[][]> arCC_Integral = new ArrayList<>();
		ArrayList<double[][]> arCC_Integral_Best = new ArrayList<>();
		
		double dClasRateBest, dClasRate, dMinFitn;
		int iMinFitn;
		
		iData = -1;
		for(int iR = 0; iR < N_rule; iR++) {
			rule = new double[sizeRule];
			
			multiPattern.clear();
			iData = rn.nextInt(nExamTrain);
			dClass = dataTrain[iData][nAttr];
			multiPattern.add(iData);
			while(multiPattern.size() < H) {
				iData = get_iData(dataTrain, dClass);
				if(!multiPattern.contains(iData)) {
					multiPattern.add(iData);
				}
			}
			
			for(int attr = 0; attr < sizeRule - 2; attr++) {
				arMinMembDegr.clear();
				for(int fs = 1; fs < nFS + 1; fs++) {
					arMembDegr.clear();
					for(int id = 0; id < multiPattern.size(); id++) {
						iData = multiPattern.get(id);
						dMembDegr = getMembershipGrade(dFuzzySets, dataTrain[iData][attr], attr, fs);
						arMembDegr.add(dMembDegr);
					}
					dMinMembDegr  = getMin(arMembDegr);
					if(dMinMembDegr >= 0.5) {
						arMinMembDegr.add(dMinMembDegr);
					} else {
						arMinMembDegr.add(0.0);
					}
				}
				if(allItems_ZeroDouble(arMinMembDegr)) {
					rule[attr] = 0;
				} else {
					rule[attr] = getIndexFuzzySet(arMinMembDegr);
				}
			}
			if(allAttr_ZeroDouble(rule) || existRuleInRB(rule, popuP)) {
				iR--;
			} else {
				rule[sizeRule - 2] = dClass;
				rule[sizeRule - 1] = rn.nextDouble();
				//System.out.println(dClass);
				popuP.add(rule);
				//showRule(rule, attr_names);
			}
		}
		
		for(int i = 0; i < q.length; i++) {
			q[i] = 1.0;
		}
		
		//showRB(popuBest, attr_names);
		
		// Calculate the fitness of genetic learning
		popuP_Best.addAll(popuP.stream().map(s->s.clone()).collect(Collectors.toList()));
		dClasRateBest = getClasRate(popuP_Best, dataTrain, dFuzzySets, sizeRule, nClas, arFitnRule, arClasiPatt, q, arCC_Integral);
		//showArrayInteger(arFitnRule);
		System.out.println("Best Classification Rate: " + dClasRateBest + " - " + showArrayDouble(q));
		
		//q
		dClasRateBest = get_CC_integral_q(dataTrain, dClasRateBest, nClas, sizeRule, q, arCC_Integral);
		System.out.println("Best Classification Rate: " + dClasRateBest + " - " + showArrayDouble(q));
		
		q_Best = q.clone();
		arCC_Integral_Best.addAll(arCC_Integral.stream().map(s->s.clone()).collect(Collectors.toList()));

		//Iterations
		int r1_aux, r2_aux, r1, r2;
		double[] rule_1;
		double[] rule_2;
		double mutaProb = 1 / nAttr;
		double mutaProbFS = 0.1;
		
		for (int iT = 0; iT < ni; iT++) {
			if((iT + 1) % 10000 == 0) {
				System.out.println("Iteration " + (iT + 1) + " of " + ni);
			}
			popuN.clear();
			
			//Genetic operators
			while(popuN.size() < N_replace / 2) {
				rule_1 = new double[sizeRule];
				rule_2 = new double[sizeRule];
				
				r1_aux = rn.nextInt(N_rule);
				r2_aux = rn.nextInt(N_rule);
				while(r2_aux == r1_aux) {
					r2_aux = rn.nextInt(N_rule);
				}
				if(arFitnRule[r1_aux] >= arFitnRule[r2_aux]) {
					r1 = r1_aux;
				} else {
					r1 = r2_aux;
				}
				r1_aux = rn.nextInt(N_rule);
				r2_aux = rn.nextInt(N_rule);
				while(r2_aux == r1_aux) {
					r2_aux = rn.nextInt(N_rule);
				}
				if(arFitnRule[r1_aux] >= arFitnRule[r2_aux]) {
					r2 = r1_aux;
				} else {
					r2 = r2_aux;
				}
			
				if(rn.nextDouble() <= crossProb) {
					for(int i = 0; i < sizeRule; i++) {
						if(rn.nextDouble() <= 0.5) {
							rule_1[i] = popuP.get(r1)[i];
							rule_2[i] = popuP.get(r2)[i];
						} else {
							rule_2[i] = popuP.get(r1)[i];
							rule_1[i] = popuP.get(r2)[i];
						}
					}
				} else {
					for(int i = 0; i < sizeRule; i++) {
						rule_1[i] = popuP.get(r1)[i];
						rule_2[i] = popuP.get(r2)[i];
					}
				}
				
				if(rn.nextDouble() <= mutaProb) {
					for(int i = 0; i < sizeRule - 2; i++) {
						if(rn.nextDouble() <= mutaProbFS) {
							rule_1[i] = rn.nextInt(nFS + 1);
						}
					}
				}
				if(rn.nextDouble() <= mutaProb) {
					for(int i = 0; i < sizeRule - 2; i++) {
						if(rn.nextDouble() <= mutaProbFS) {
							rule_2[i] = rn.nextInt(nFS + 1);
						}
					}
				}
				
				if(!existRuleInRB(rule_1, popuP)) {
					popuN.add(rule_1);
				}
				if(popuN.size() >= N_replace / 2) {
					break;
				}
				if(!existRuleInRB(rule_2, popuP)) {
					popuN.add(rule_2);
				}
				
			}
			
			while(popuN.size() < N_replace) {
				rule = new double[sizeRule];
				
				multiPattern.clear();
				iData = getBasePatt(arClasiPatt);
				dClass = dataTrain[iData][nAttr];
				multiPattern.add(iData);
				while(multiPattern.size() < H) {
					iData = get_iData(dataTrain, dClass);
					if(!multiPattern.contains(iData)) {
						multiPattern.add(iData);
					}
				}
				
				for(int attr = 0; attr < sizeRule - 2; attr++) {
					arMinMembDegr.clear();
					for(int fs = 1; fs < nFS + 1; fs++) {
						arMembDegr.clear();
						for(int id = 0; id < multiPattern.size(); id++) {
							iData = multiPattern.get(id);
							dMembDegr = getMembershipGrade(dFuzzySets, dataTrain[iData][attr], attr, fs);
							arMembDegr.add(dMembDegr);
						}
						dMinMembDegr  = getMin(arMembDegr);
						if(dMinMembDegr >= 0.5) {
							arMinMembDegr.add(dMinMembDegr);
						} else {
							arMinMembDegr.add(0.0);
						}
					}
					if(allItems_ZeroDouble(arMinMembDegr)) {
						rule[attr] = 0;
					} else {
						rule[attr] = getIndexFuzzySet(arMinMembDegr);
					}
				}
				if(!allAttr_ZeroDouble(rule)) {
					if(!existRuleInRB(rule, popuP)) {
						rule[sizeRule - 2] = (int) dClass;
						rule[sizeRule - 1] = rn.nextDouble();
						popuN.add(rule);
					}
				}
			}
			
			//4-2
			worst_rules.clear();
			while(worst_rules.size() != N_replace) {
				dMinFitn = Double.MAX_VALUE;
				iMinFitn = -1;
				for(int i = 0; i < arFitnRule.length; i++) {
					if(!worst_rules.contains(i)) {
						if(arFitnRule[i] < dMinFitn) {
							dMinFitn = arFitnRule[i];
							iMinFitn = i;
						}
					}
				}
				worst_rules.add(iMinFitn);
			}

			for(int i = 0; i < worst_rules.size(); i++) {
				for(int i1 = 0; i1 < sizeRule; i1++) {
					popuP.get(worst_rules.get(i))[i1] = popuN.get(i)[i1];
				}
			}
			
			//4-3
			dClasRate = getClasRate(popuP, dataTrain, dFuzzySets, sizeRule, nClas, arFitnRule, arClasiPatt, q, arCC_Integral);
			dClasRate = get_CC_integral_q(dataTrain, dClasRateBest, nClas, sizeRule, q, arCC_Integral);
			
			//4-4
			if(dClasRate > dClasRateBest) {
				q_Best = q.clone();
				arCC_Integral_Best.clear();
				arCC_Integral_Best.addAll(arCC_Integral.stream().map(s->s.clone()).collect(Collectors.toList()));
				popuP_Best.clear();
				popuP_Best.addAll(popuP.stream().map(s->s.clone()).collect(Collectors.toList()));
				dClasRateBest = dClasRate;
			} else {
				q = q_Best.clone();
				arCC_Integral.clear();
				arCC_Integral.addAll(arCC_Integral_Best.stream().map(s->s.clone()).collect(Collectors.toList()));
				popuP.clear();
				popuP.addAll(popuP_Best.stream().map(s->s.clone()).collect(Collectors.toList()));
			}
			
			if((iT + 1) % 10000 == 0) {
				System.out.println("\tBest Classification Rate: " + dClasRateBest + " - " + showArrayDouble(q_Best));
			}
		}
		
		System.out.println();
		showRB(popuP_Best, attr_names);
		System.out.println();
		
		double dClasAccuTest = getClasRateTest(popuP_Best, dataTest, dFuzzySets, sizeRule, nClas, q_Best);
		System.out.println("Classification Rate Train: " + dClasRateBest);
		System.out.println("Classification Rate Test: " + dClasAccuTest);
		System.out.println("Michigan EE CC integral");
		System.out.println(sDS + " - " + iCV + " - " + iFold + ": " + dClasRateBest + "\t" + dClasAccuTest);
		System.out.println();
		createFileResults(sDS, iCV, iFold, iResult, dClasRateBest, dClasAccuTest);
		System.out.println("**** END ****");
	}
	
	private static void createFileResults(String sDS, int iCV, int iFold, int iResult, double dClasRateBest, double dClasAccuTest) throws FileNotFoundException, UnsupportedEncodingException {
		String sFileName = System.getProperty("user.dir") + File.separator + "results" + File.separator + sDS + "-" + iCV + "-fold" +  File.separator + sDS + "_" + iCV + "_" + iFold + "_" + iResult + ".txt"; 
		PrintWriter writer = new PrintWriter(sFileName);
		writer.println(dClasRateBest + "\t" + dClasAccuTest);
		writer.close();
	}

	private static double get_CC_integral_q(double[][] Ddata, double dClasRateBest, int nClass, int sizeRule, double[] arCChoquet, ArrayList<double[][]> arCC_Integral) {
		int iMaxTM;
		double iCorrect, valu;
		double dMaxTM, TM, size;
		Random rn = new Random();
		double[][] CC_integral;
		double[] arCChoquet_Best = new double[nClass];
		ArrayList<Double> arAssoDegr = new ArrayList<>();
		ArrayList<Double> arTM = new ArrayList<>();
		
		for(int i = 0; i < arCChoquet.length; i++) {
			arCChoquet_Best[i] = arCChoquet[i];
		}
		
		for(int cc = 0; cc < 5; cc++) {
			iCorrect = 0.0;
			int prob = (rn.nextInt(5) + 1) * 10;
			for(int i = 0; i < nClass; i++) {
				if(rn.nextInt(100) < prob) {
					valu = 0.01 + (1.99 - 0.01) * rn.nextDouble();
					if(valu <= 1.0) {
						arCChoquet[i] = valu;
					} else {
						arCChoquet[i] = 1 / (2 - valu);
					}
				}
			}
			for (int ex = 0; ex < Ddata.length; ex++) {
				CC_integral = arCC_Integral.get(ex);
				arTM.clear();
				for(int c = 0; c < nClass; c++) {
					arAssoDegr.clear();
					for(int f = 0; f < CC_integral.length; f++) {
						if(CC_integral[f][c] != 0.0) {
							arAssoDegr.add(CC_integral[f][c]);
						}
					}
					if(arAssoDegr.isEmpty()) {
						arTM.add(0.0);
					} else {
						arAssoDegr.sort(Comparator.naturalOrder());
						size = (double) arAssoDegr.size();
						TM = Math.min(arAssoDegr.get(0), size / size) - Math.min(0, size / size);
						for(int i = 1; i < arAssoDegr.size(); i++) {
							TM = TM + Math.min(arAssoDegr.get(i), (size - i) / size) - Math.min(arAssoDegr.get(i - 1), (size - i) / size);
						}
						arTM.add(Math.pow(TM, arCChoquet[c]));
					}
				}
				dMaxTM = getMax(arTM);
				if(dMaxTM > 0.0) {
					iMaxTM = getMaxIndex(arTM);
					if(iMaxTM + 1 == (int) Ddata[ex][sizeRule - 2]) {
						iCorrect++;
					}
				}
			}
			if(iCorrect / Ddata.length > dClasRateBest) {
				dClasRateBest = iCorrect / Ddata.length;
				for(int i = 0; i < arCChoquet.length; i++) {
					arCChoquet_Best[i] = arCChoquet[i];
				}
			} else {
				for(int i = 0; i < arCChoquet.length; i++) {
					arCChoquet[i] = arCChoquet_Best[i];
				}
			}
		}
		
		for(int i = 0; i < arCChoquet.length; i++) {
			arCChoquet[i] = arCChoquet_Best[i];
		}
		
		return dClasRateBest;
	}

	private static boolean existRuleInRB(double[] rule, ArrayList<double[]> popuP) {
		double[] rule_aux;
		boolean bEqual = true;
		if(popuP.size() == 0) {
			return false;
		}
		for(int i = 0; i < popuP.size(); i++) {
			bEqual = true;
			rule_aux = popuP.get(i);
			for(int r = 0; r < rule_aux.length - 2; r++) {
				if(rule[r] != rule_aux[r]) {
					bEqual = false;
					break;
				}
			}
			if(bEqual) {
				break;
			}
		}
		return bEqual;
	}
	
	private static int getBasePatt(boolean[] ar) {
		// TODO Auto-generated method stub
		Random rn = new Random();
		boolean allClasi = true;
		int aux;
		for(int i = 0; i< ar.length; i++){
			if(ar[i] == false) {
				allClasi = false;
				break;
			}
		}
		if(allClasi) {
			return rn.nextInt(ar.length);
		} else {
			while(true) {
				aux = rn.nextInt(ar.length);
				if(ar[aux] == false) {
					return aux;
				}
			}
		}
	}

	private static boolean allAttr_ZeroInt(int[] ar) {
		for(int i = 0; i< ar.length; i++){
			if(ar[i] != 0) {
				return false;
			}
		}
		return true;
	}
	
	private static boolean allAttr_ZeroDouble(double[] ar) {
		for(int i = 0; i< ar.length - 2; i++){
			if(ar[i] != 0.0) {
				return false;
			}
		}
		return true;
	}

	private static int getIndexFuzzySet(ArrayList<Double> inputArray) {
		double sum = 0.0;
		double random;
		int index = -1;
		Random rn = new Random();
		ArrayList<Double> prob = new ArrayList<>();
		ArrayList<Double> sum_prob = new ArrayList<>();
		for(int i = 0; i< inputArray.size(); i++){
			sum = sum + inputArray.get(i);
		}
		for(int i = 0; i< inputArray.size(); i++){
			prob.add(inputArray.get(i) / sum);
		}
		sum_prob.add(prob.get(0));
		for(int i = 1; i < prob.size(); i++){
			sum_prob.add(sum_prob.get(i - 1) + prob.get(i));
		}
		random = rn.nextDouble();
		if(random <= sum_prob.get(0)) {
			return 1;
		}
		for(int i = 0; i < prob.size() - 2; i++){
			if(random > sum_prob.get(i) && random <= sum_prob.get(i + 1)) {
				return i + 2;
			}
		}
		if(random > sum_prob.get(sum_prob.size() - 2)) {
			return sum_prob.size();
		}
		return index;
	}

	private static boolean allItems_ZeroDouble(ArrayList<Double> inputArray) {
		for(int i = 0; i<inputArray.size(); i++){
			if(inputArray.get(i) != 0.0) {
				return false;
			}
		}
		return true;
	}

	private static boolean attrInRule(int[] rule, int attr, int sizeRule) {
		int aux;
		for(int r = 0; r < sizeRule; r = r + 2) {
			aux = rule[r];
			if(aux == attr) {
				return true;
			}
		}
		return false;
	}

	private static void showRule(double[] rule, String[] attr_names) {
		StringBuilder sRule = new StringBuilder();
		int fs;
		sRule.append("If ");
		for (int i = 0; i < rule.length - 2; i++) {
			fs = (int) rule[i];
			if(fs != 0.0) {
				sRule.append(attr_names[i] + " is " + getLingTerm(fs));
				sRule.append(" and ");
			}
		}
		if(sRule.length() > 5) {
			sRule.setLength(sRule.length() - 5);
		}
		sRule.append(" then Class " + (int) rule[rule.length - 2] + " with " + rule[rule.length - 1]); 
		System.out.println(sRule.toString());
	}

	private static int get_iData(double[][] dataTrain, double dClass) {
		Random rn = new Random();
		int iData = rn.nextInt(dataTrain.length);
		while(dataTrain[iData][dataTrain[iData].length - 1] != dClass) {
			iData = rn.nextInt(dataTrain.length);
		}
		return iData;
	}

	private static double getFitnRule(int[] rule, double[][] Ddata, double[][] dFuzzySets, int inputs, int Ioutputs, int sizeRule, double[] dataWeig) {
		double dMembDegr, dFireDegr;
		ArrayList<Double> valuesAnd = new ArrayList<>();
		ArrayList<Double> valuesOr = new ArrayList<>();
		double dNega = 0.0;
		double dPosi = 0.0;
		int iPosiClas = sizeRule * 2;
		int iRuleClas;
		boolean bAllDontCare = true;
		int attr, fuzzySet, oper;

		for (int i = 0; i < inputs; i++) {
			if (rule[i] != 0.0) {
				bAllDontCare = false;
				break;
			}
		}
		if (bAllDontCare) {
			return Double.MAX_VALUE * -1;
		}
		for (int ex = 0; ex < Ddata.length; ex++) {
			for (int r = 0; r < sizeRule; r = r + 2) {
				fuzzySet = rule[r + 1];
				if(fuzzySet != 0) {
					attr = rule[r];
					if (Ddata[ex][attr] == 0.0) {
						valuesAnd.clear();
						break;
					} else {
						dMembDegr = getMembershipGrade(dFuzzySets, Ddata[ex][attr], attr, fuzzySet);
						dMembDegr = dMembDegr * dataWeig[ex];
						valuesAnd.add(dMembDegr);
					}
				}
			}
			if (valuesAnd.size() == 0) {
				dFireDegr = 0.0;
			} else {
				dFireDegr = getMin(valuesAnd);
			}
			valuesAnd.clear();
			iRuleClas = rule[iPosiClas];
			if ((int) Ddata[ex][inputs] == iRuleClas) {
				dPosi = dPosi + dFireDegr;
			} else {
				dNega = dNega + dFireDegr;
			}
		}
		return dPosi - dNega;
	}

	private static double getClasRate(ArrayList<double[]> RB, double[][] Ddata, double[][] dFuzzySets, int sizeRule, int nClass, double[] arFitnRule, boolean[] arClasiPatt, double[] arCChoquet, ArrayList<double[][]> arCC_Integral) {
		int iCorrect = 0;
		double dMembDegr;
		double[] rule;

		int fuzzySet,iMaxTM, classRule;
		double fireDegr, dMaxTM, TM, size;
		ArrayList<Double> arAssoDegr = new ArrayList<>();
		ArrayList<Double> arTM = new ArrayList<>();
		ArrayList<Double> ardPosi = new ArrayList<>();
		ArrayList<Double> ariPosi = new ArrayList<>();
		ArrayList<Double> ardNega = new ArrayList<>();
		ArrayList<Double> ariNega = new ArrayList<>();
		
		for(int i = 0; i < arClasiPatt.length; i++) {
			arClasiPatt[i] = false;
		}
		
		for(int i = 0; i < arFitnRule.length; i++) {
			arFitnRule[i] = 0.0;
			ardPosi.add(0.0);
			ariPosi.add(0.0);
			ardNega.add(0.0);
			ariNega.add(0.0);
		}
		
		arCC_Integral.clear();
		
		for (int ex = 0; ex < Ddata.length; ex++) {
			double[][] CC_integral = new double[RB.size()][nClass];

			for (int i = 0; i < RB.size(); i++) {
				rule = RB.get(i);
				fireDegr = 1.0;
				for(int attr = 0; attr < sizeRule - 2; attr++) {
					fuzzySet = (int) rule[attr];
					if(fuzzySet > 0) {
						dMembDegr = getMembershipGrade(dFuzzySets, Ddata[ex][attr], attr, fuzzySet);
						fireDegr = fireDegr * dMembDegr;
						if(fireDegr == 0.0) {
							break;
						}
					}
				}
				if(fireDegr > 0.0) {
					classRule = (int) rule[sizeRule - 2];
					CC_integral[i][classRule - 1] = fireDegr * rule[sizeRule - 1];
					if(classRule == Ddata[ex][sizeRule - 2]) {
						ardPosi.set(i, ardPosi.get(i) + fireDegr * rule[sizeRule - 1]);
						ariPosi.set(i, ariPosi.get(i) + 1.0);
					} else {
						ardNega.set(i, ardNega.get(i) + fireDegr * rule[sizeRule - 1]);
						ariNega.set(i, ariNega.get(i) + 1.0);
					}
				}
			}
			
			arCC_Integral.add(CC_integral);
			
		}
		
		double[][] CC_integral;
		for (int ex = 0; ex < Ddata.length; ex++) {
			CC_integral = arCC_Integral.get(ex);
			arTM.clear();
			for(int c = 0; c < nClass; c++) {
				arAssoDegr.clear();
				for(int f = 0; f < RB.size(); f++) {
					if(CC_integral[f][c] != 0.0) {
						arAssoDegr.add(CC_integral[f][c]);
					}
				}
				if(arAssoDegr.isEmpty()) {
					arTM.add(0.0);
				} else {
					arAssoDegr.sort(Comparator.naturalOrder());
					size = (double) arAssoDegr.size();
					TM = Math.min(arAssoDegr.get(0), size / size) - Math.min(0, size / size);
					for(int i = 1; i < arAssoDegr.size(); i++) {
						TM = TM + Math.min(arAssoDegr.get(i), (size - i) / size) - Math.min(arAssoDegr.get(i - 1), (size - i) / size);
					}
					arTM.add(Math.pow(TM, arCChoquet[c]));
				}
			}
			dMaxTM = getMax(arTM);
			if(dMaxTM > 0.0) {
				iMaxTM = getMaxIndex(arTM);
				if(iMaxTM + 1 == (int) Ddata[ex][sizeRule - 2]) {
					iCorrect++;
				}
			}
		}
	
		//Rule fitness
		for(int i = 0; i < arFitnRule.length; i++) {
			if(ariPosi.get(i) > 0 && ariNega.get(i) > 0) {
				arFitnRule[i] = ardPosi.get(i) / ariPosi.get(i) - ardNega.get(i) * 2 / ariNega.get(i);
			} else if (ariPosi.get(i) > 0) {
				arFitnRule[i] = ardPosi.get(i) / ariPosi.get(i);
			} else if (ariNega.get(i) > 0) {
				arFitnRule[i] = - ardNega.get(i) * 2 / ariNega.get(i);
			}
		}
	
		return (double) iCorrect / Ddata.length;
	}
	
	private static double getClasRateTest(ArrayList<double[]> RB, double[][] Ddata, double[][] dFuzzySets, int sizeRule, int nClass, double[] arCChoquet) {
		int iCorrect = 0;
		double dMembDegr;
		double[] rule;
		int fuzzySet,iMaxTM, classRule;
		double fireDegr, dMaxTM, TM, size;
		ArrayList<double[][]> arCC_Integral = new ArrayList<>();
		ArrayList<Double> arAssoDegr = new ArrayList<>();
		ArrayList<Double> arTM = new ArrayList<>();
		
		for (int ex = 0; ex < Ddata.length; ex++) {
			double[][] CC_integral = new double[RB.size()][nClass];

			for (int i = 0; i < RB.size(); i++) {
				rule = RB.get(i);
				fireDegr = 1.0;
				for(int attr = 0; attr < sizeRule - 2; attr++) {
					fuzzySet = (int) rule[attr];
					if(fuzzySet > 0) {
						dMembDegr = getMembershipGrade(dFuzzySets, Ddata[ex][attr], attr, fuzzySet);
						fireDegr = fireDegr * dMembDegr;
						if(fireDegr == 0.0) {
							break;
						}
					}
				}
				if(fireDegr > 0.0) {
					classRule = (int) rule[sizeRule - 2];
					CC_integral[i][classRule - 1] = fireDegr * rule[sizeRule - 1];
				}
			}
			
			arCC_Integral.add(CC_integral);
			
		}
		
		double[][] CC_integral;
		for (int ex = 0; ex < Ddata.length; ex++) {
			CC_integral = arCC_Integral.get(ex);
			arTM.clear();
			for(int c = 0; c < nClass; c++) {
				arAssoDegr.clear();
				for(int f = 0; f < RB.size(); f++) {
					if(CC_integral[f][c] != 0.0) {
						arAssoDegr.add(CC_integral[f][c]);
					}
				}
				if(arAssoDegr.isEmpty()) {
					arTM.add(0.0);
				} else {
					arAssoDegr.sort(Comparator.naturalOrder());
					size = (double) arAssoDegr.size();
					TM = Math.min(arAssoDegr.get(0), size / size) - Math.min(0, size / size);
					for(int i = 1; i < arAssoDegr.size(); i++) {
						TM = TM + Math.min(arAssoDegr.get(i), (size - i) / size) - Math.min(arAssoDegr.get(i - 1), (size - i) / size);
					}
					arTM.add(Math.pow(TM, arCChoquet[c]));
				}
			}
			dMaxTM = getMax(arTM);
			if(dMaxTM > 0.0) {
				iMaxTM = getMaxIndex(arTM);
				if(iMaxTM + 1 == (int) Ddata[ex][sizeRule - 2]) {
					iCorrect++;
				}
			}
		}
	
		return (double) iCorrect / Ddata.length;
	}
	
	private static void showResultProm(double[] promClas,  int[] aPredClass, int[] aRealClas, double dTH, boolean bStat) {
		for(int i = 0; i < promClas.length; i++) {
			System.out.print(promClas[i] + " ");
		}
		System.out.print(" - ");
		for(int i = 0; i < aPredClass.length; i++) {
			System.out.print(aPredClass[i]);
		}
		System.out.print(" - ");
		System.out.print(dTH);
		System.out.print(" - ");
		for(int i = 0; i < aRealClas.length; i++) {
			System.out.print(aRealClas[i]);
		}
		if(bStat) {
			System.out.print(" - C");
		}
		else {
			System.out.print(" - E");
		}
		System.out.println();
	}

	private static void showResult(int[] aPredClass, int[] aRealClas, boolean bEqua) {
		for(int i = 0; i < aPredClass.length; i++) {
			System.out.print(aPredClass[i]);
		}
		System.out.print(" - ");
		for(int i = 0; i < aRealClas.length; i++) {
			System.out.print(aRealClas[i]);
		}
		if(bEqua) {
			System.out.println(" - C");
		}
		else {
			System.out.println(" - E");
		}
	}

	private static void showRB(ArrayList<double[]> RB, String[] attr_names) {
		for(int iRB = 0; iRB < RB.size(); iRB++) {
			showRule(RB.get(iRB), attr_names);
		}
	}
	
	private static String getLingTerm(int lt) {
		String sLingTerm = "?";

		switch (lt) {
		case 0:
			sLingTerm = "Don't Care";
			break;
		case 1:
			sLingTerm = "VERY LOW";
			break;
		case 2:
			sLingTerm = "LOW";
			break;
		case 3:
			sLingTerm = "MEDIUM";
			break;
		case 4:
			sLingTerm = "HIGH";
			break;
		case 5:
			sLingTerm = "VERY HIGH";
			break;
		}

		return sLingTerm;
	}

	private static double getMembershipGrade(double[][] dFuzzySets, double x, int iLingTerm, int iFuzzySet) {
		double a;
		double b;
		double c;

		iFuzzySet--;
		a = dFuzzySets[iLingTerm][iFuzzySet * 3];
		b = dFuzzySets[iLingTerm][iFuzzySet * 3 + 1];
		c = dFuzzySets[iLingTerm][iFuzzySet * 3 + 2];

		//System.out.println(a + "\t" + b + "\t" + c);

		if (x <= a) {
			return 0.0;
		}
		if (x > a && x <= b) {
			return (x - a) / (b - a);
		}
		if (x > b && x < c) {
			return (c - x) / (c - b);
		}
		if (x >= c) {
			return 0.0;
		}

		return 0.0;
	}

	private static void showMatrizDouble(double[][] matrix) {
		for (int r = 0; r < matrix.length; r++) {
			for (int c = 0; c < matrix[r].length; c++) {
				System.out.print(matrix[r][c] + "\t");
			}
			System.out.println();
		}
	}

	private static void showMatrizInteger(int[][] matrix) {
		for (int r = 0; r < matrix.length; r++) {
			for (int c = 0; c < matrix[r].length; c++) {
				System.out.print(matrix[r][c] + "\t");
			}
			System.out.println();
		}
	}
	
	private static String showArrayDouble(double[] ar) {
		StringBuilder s = new StringBuilder();
		for (int i = 0; i < ar.length; i++) {
			s.append(ar[i] + " ");
		}
		return s.toString();
	}
	
	public static double getMin(ArrayList<Double> inputArray){ 
	    double minValue = inputArray.get(0); 
	    for(int i = 1; i< inputArray.size(); i++){
	    	if(inputArray.get(i) < minValue){ 
	    		minValue = inputArray.get(i); 
	    	} 
	    } 
	    return minValue; 
	 }
	 
	 public static int getMinIndex(ArrayList<Double> inputArray){ 
	    double minValue = inputArray.get(0);
	    int minIndex = 0;
	    for(int i = 1; i< inputArray.size(); i++){
	    	if(inputArray.get(i) < minValue){ 
	    		minValue = inputArray.get(i);
	    		minIndex = i;
	    	} 
	    } 
	    return minIndex; 
	 }
	 
	 public static double getMax(ArrayList<Double> inputArray) { 
		 double maxValue = inputArray.get(0); 
		 for(int i = 1; i<inputArray.size(); i++){
		     if(inputArray.get(i) > maxValue) {
		    	 maxValue = inputArray.get(i);
		     }
		 }
	     return maxValue; 
	 }
	 
	 public static int getMaxIndex(ArrayList<Double> inputArray){ 
	    double maxValue = inputArray.get(0);
	    int maxIndex = 0;
	    for(int i = 1; i< inputArray.size(); i++){
	    	if(inputArray.get(i) > maxValue){ 
	    		maxValue = inputArray.get(i);
	    		maxIndex = i;
	    	} 
	    } 
	    return maxIndex; 
	 }
}