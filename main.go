package main

import (
	"bufio"
	"encoding/csv"
	"encoding/json"
	"hector/algo"
	"io"
	"net/http"
	"os"

	"github.com/bmizerany/pat"
	"github.com/codegangsta/negroni"
	"github.com/xlvector/hector/util"

	"strconv"

	"log"

	"github.com/xlvector/hector"
	"github.com/xlvector/hector/core"
)

// NewServer - init http server
func NewServer() *http.Server {
	n := negroni.New()
	n.Use(negroni.NewLogger())
	// Setup routes
	router := pat.New()
	router.Get("/api/predict/:id", http.HandlerFunc(predict))
	// Add the router action
	n.UseHandler(router)
	Server := &http.Server{
		Addr:           ":8080",
		Handler:        n,
		MaxHeaderBytes: 1 << 20,
	}
	return Server
}

func predict(res http.ResponseWriter, req *http.Request) {
	driverID, _ := strconv.Atoi(req.URL.Query().Get(":id"))
	var model algo.Classifier
	if _, ok := driversModels[driverID]; !ok {
		model = NewModel(driverID)
	} else {
		model = driversModels[driverID]
	}
	fs := make(map[string]float64)
	fs["hour"], _ = strconv.ParseFloat(req.URL.Query().Get("hour"), 64)
	fs["dayOfWeek"], _ = strconv.ParseFloat(req.URL.Query().Get("dayOfWeek"), 64)
	fs["distance_from_order_on_creation"], _ = strconv.ParseFloat(req.URL.Query().Get("distance_from_order_on_creation"), 64)
	fs["driver_location_key"], _ = strconv.ParseFloat(req.URL.Query().Get("driver_location_key"), 64)
	fs["driver_latitude"], _ = strconv.ParseFloat(req.URL.Query().Get("driver_latitude"), 64)
	fs["driver_longitude"], _ = strconv.ParseFloat(req.URL.Query().Get("driver_longitude"), 64)
	fs["origin_location_key"], _ = strconv.ParseFloat(req.URL.Query().Get("origin_location_key"), 64)
	fs["origin_latitude"], _ = strconv.ParseFloat(req.URL.Query().Get("origin_latitude"), 64)
	fs["origin_longitude"], _ = strconv.ParseFloat(req.URL.Query().Get("origin_longitude"), 64)
	sample := NewSample(fs)
	pr := model.Predict(sample)
	renderJSON(res, http.StatusOK, map[string]interface{}{"predict": pr})
}

func renderJSON(r http.ResponseWriter, status int, v interface{}) {
	var result []byte
	var err error
	result, err = json.Marshal(v)
	if err != nil {
		http.Error(r, err.Error(), 500)
		return
	}
	// json rendered fine, write out the result
	r.Header().Set("Content-Type", "application/json")
	r.WriteHeader(status)
	r.Write(result)
}

var driversModels map[int]algo.Classifier

func main() {
	driversModels = make(map[int]algo.Classifier)
	server := NewServer()
	log.Printf("Start serving on %s", server.Addr)
	log.Println(server.ListenAndServe())
}

func SplitFile(dataset *core.DataSet, total, part int) (*core.DataSet, *core.DataSet) {
	train := core.NewDataSet()
	test := core.NewDataSet()
	for i, sample := range dataset.Samples {
		if i%total == part {
			test.AddSample(sample)
		} else {
			train.AddSample(sample)
		}
	}
	return train, test
}

func NewSample(fs map[string]float64) *core.Sample {
	sample := core.NewSample()
	for k, v := range fs {
		f := core.Feature{
			Id:    util.Hash(k),
			Value: v,
		}
		sample.AddFeature(f)
	}
	return sample
}

func NewModel(driversID int) algo.Classifier {
	params := make(map[string]string)
	params["steps"] = "30"
	params["max-depth"] = "7"
	params["min-leaf-size"] = "10"
	params["tree-count"] = "10"
	params["learning-rate"] = "0.0001"
	params["learning-rate-discount"] = "1.0"
	params["regularization"] = "0.0001"
	params["gini"] = "1.0"
	params["hidden"] = "15"
	params["k"] = "10"
	params["feature-count"] = "9.0"
	params["dt-sample-ratio"] = "1.0"
	driversModels[driversID] = hector.GetClassifier("rf")
	dataSet, _ := NewDataSetSample("716new.csv")
	//dataSetTest, statusID := NewDataSetSample("716dec-3.csv")
	driversModels[driversID].Init(params)
	log.Println("Train")

	//train, test := SplitFile(dataSet, 2, 0)
	//auc, _ := hector.AlgorithmRunOnDataSet(driversModels[driversID[i]], train, test, "", params)
	//log.Print("AUC: ")
	//log.Println(auc)

	driversModels[driversID].Train(dataSet)
	log.Println("save")
	driversModels[driversID].SaveModel("716")
	return driversModels[driversID]
	//driversModels[driversID[i]].LoadModel("6116")

	/*log.Println("predict... ")
	//sample := SampleTest()
	predictAccept := 0.0
	predictAcceptTotal := 0.0
	predictReject := 0.0
	predictRejectTotal := 0.0
	countR := 0
	countA := 0
	for j := 0; j < len(dataSetTest.Samples); j++ {
		sample := dataSetTest.Samples[j]
		predict := driversModels[driversID[i]].Predict(sample)
		if statusID[j] == "4" {
			predictAcceptTotal++
			predictAccept += predict
			if predict > 0.7 {
				countA++
				log.Println("A:", predict)
			}

		} else {
			predictReject += predict
			predictRejectTotal++
			if predict > 0.7 {
				countR++
				log.Println("R:", predict)
			}
		}
	}
	log.Print("Accept: ")
	log.Println(predictAccept / predictAcceptTotal)
	log.Println(countA)
	log.Print("Reject: ")
	log.Println(predictReject / predictRejectTotal)
	log.Println(countR)*/
}

func NewDataSetSample(path string) (*core.DataSet, []string) {
	samples := []*core.Sample{}
	statusID := []string{}
	f, _ := os.Open(path)
	r := csv.NewReader(bufio.NewReader(f))
	r.Read()
	for {
		record, err := r.Read()
		// Stop at EOF.
		if err == io.EOF {
			break
		}
		sample := core.NewSample()
		fs := make(map[string]float64)
		log.Println(record[2])
		statusID = append(statusID, record[2])
		switch record[2] {
		case "4", "2":
			sample.Label = 1
		default:
			sample.Label = 0
		}
		log.Println(record[10])
		if sample.Label == 0 && record[10] != "" && record[2] != "3" {
			sample.Label = 1
		}
		log.Println("hour " + record[4])
		fs["hour"], _ = strconv.ParseFloat(record[4], 64)

		log.Println("day_of_week " + record[5])
		fs["day_of_week"], _ = strconv.ParseFloat(record[5], 64)

		log.Println("distance_from_order_on_creation " + record[6])
		fs["distance_from_order_on_creation"], _ = strconv.ParseFloat(record[6], 64)

		log.Println("driver_location_key " + record[7])
		fs["driver_location_key"], _ = strconv.ParseFloat(record[7], 64)

		log.Println("driver_latitude " + record[8])
		fs["driver_latitude"], _ = strconv.ParseFloat(record[8], 64)

		log.Println("driver_longitude " + record[9])
		fs["driver_longitude"], _ = strconv.ParseFloat(record[9], 64)

		log.Println("origin_location_key " + record[13])
		fs["origin_location_key"], _ = strconv.ParseFloat(record[13], 64)

		log.Println("origin_latitude " + record[14])
		fs["origin_latitude"], _ = strconv.ParseFloat(record[14], 64)

		log.Println("origin_longitude " + record[15])
		fs["origin_longitude"], _ = strconv.ParseFloat(record[15], 64)

		for k, v := range fs {
			f := core.Feature{
				Id:    util.Hash(k),
				Value: v,
			}
			sample.AddFeature(f)
		}
		samples = append(samples, sample)
	}
	d := &core.DataSet{
		Samples: samples,
	}
	return d, statusID
}
