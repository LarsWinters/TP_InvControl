from scipy.stats import norm

def check_sl_input(SL):
    try:
        #convert to int
        val = int(SL)
        print("Service Level goal is " + SL + "%.")
    except ValueError:
        try:
            #convert to float
            val = float(SL)
            print("Service Level goal is " + "{:.1%}".format(val))
        except ValueError:
            print("Please input either the percentage value (e.g. 95) or a float value (e.g. 0.95).")
            main()

def format_sl_input(service_level):
    try:
        val = int(service_level)
        val = val / 100
        return val
    except ValueError:
        try:
            val = float(service_level)
            return val
        except ValueError:
            print("Error")

def check_mean_demand(mean_demand):
    try:
        val = int(mean_demand)
        return val
    except ValueError:
        print("Error: check_mean_demand")

def check_leadtime(lead_time):
    try:
        val = int(lead_time)
        return val
    except ValueError:
        print("Error: check_leadtime")



def main():
    print("Safety Stock Calculation\n")
    print("\n")
    service_level = input("Please input the service level goal (e.g.: 0.95 or 95): ")
    check_sl_input(service_level)
    service_level = format_sl_input(service_level)
    print("Successfully formatted", service_level)
    mean_demand = input("Please input the mean demand during lead time Z: ")
    check_mean_demand(mean_demand)
    lead_time = input("Please input the mean leadtime Z in days: ")
    check_leadtime(lead_time)
    service_factor = norm.ppf(service_level)
    result = float(service_factor) * math.sqrt(float(mean_demand) * float(lead_time))
    print(int(result))

if __name__ == "__main__":
    main()