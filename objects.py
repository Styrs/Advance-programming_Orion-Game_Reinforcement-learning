import random
import math

#################################################################################################################################
## ------------------------------------------------------------------------------------------------------------------------------
## Planet class
## ------------------------------------------------------------------------------------------------------------------------------
#################################################################################################################################

class planets_Moon:

    def __init__(self,distance,star,x,y,type = None,name = None,masse = None,radius = None,surface_kelvin =None):
        self.distance = distance
        self.type = self.type = type if type is not None else self.random_type_planet()
        self.category = "Planetoid"
        self.positionX = x
        self.positionY = y
        self.name = name if name is not None else f"No names"
        self.masse = masse if masse is not None else self.random_masse_planet() #in earth's mass
        self.color = self.compute_planet_color()

        self.radius = radius if radius is not None else self.random_radius_planet() #in earth's radius
        self.radius_pixel = self.compute_radius_pixel()
        self.solar_exposition = self.compute_solar_exposition(star)
        self.surface_gravity = self.compute_surface_gravity()
        self.surface = self.compute_surface_area()
        self.surface_construct = self.surface
        self.surface_kelvin = self.surface_kelvin = surface_kelvin if surface_kelvin is not None else self.random_surface_temperature()
        
        self.facteur_construction_gravity = None
        self.facteur_construction_temperature = None

        self.facteur_construction = self.initialisation_construction_factor ()



## ----------------------------------------------------------------------------------------------------------------------
## Definition of deterministic characteristics 
## ----------------------------------------------------------------------------------------------------------------------

    def compute_planet_color (self):
            
            if self.type == "Telluric":
                
                return (180,150,165)
            elif self.type == "Gaz Giant":
                return (235,220,100)


    def compute_solar_exposition (self,star):
            c = 5.67 * (10)**-8
            solar_radius = star.radius * 6.9634 * (10**8)
            distance_km = self.distance * 1.496 * (10**11)
            E = c*(star.kelvin**4)*((solar_radius/distance_km)**2)
            return(E)


    def compute_radius_pixel (self):
        if self.type == "Telluric":
            return self.radius * 2.5
        elif self.type == "Gaz Giant":
            return self.radius * 1.5


    def compute_surface_gravity(self):
        """Compute surface gravity in m/s² based on mass and radius (Earth units)."""
        g0 = 9.81  # m/s² (Earth gravity)
        return g0 * (self.masse / (self.radius ** 2))
    

    def compute_surface_area(self):
        """Compute the planet's surface area in km²."""
        earth_radius_km = 6371
        return 4 * math.pi * (self.radius * earth_radius_km) ** 2


## ----------------------------------------------------------------------------------------------------------------------
## Definition of random characteristics 
## ----------------------------------------------------------------------------------------------------------------------



    def random_type_planet(self):
        pourcentage = random.randint(1,100)
        if self.distance <= 3:
            if pourcentage > 70:
                return "Gaz Giant" #Meaning it is a gaz giant 
            elif pourcentage <= 70:
                return "Telluric" 
        if self.distance > 3:
            if pourcentage > 30:
                return "Gaz Giant" #Meaning it is a gaz giant 
            elif pourcentage <= 30:
                return "Telluric" 


    def random_masse_planet (self):
        if self.type == "Telluric":
            masse = random.randint(1,50)/10
            return masse
        elif self.type == "Gaz Giant":
            masse = random.randint(10,300)
            return masse


    def random_radius_planet (self):

        if self.type == "Telluric":
            a = 0.5
            b = 0.3
            var = 0.08**2
            r_min = 0.4
            r_max = 2.2
        if self.type == "Gaz Giant":
            a = 3.6
            b = 0.023
            var = 0.8**2
            r_min = 2.5
            r_max = 16
        noise = random.normalvariate(0,var**0.5)
        r = a + b*self.masse + noise
        r = max(r_min, min(r, r_max))
        return(r)
    

    def random_surface_temperature (self):
        if self.type == "Telluric":
            A = random.uniform(0.15, 0.60)
            c = 5.67 * (10)**-8
            kelvin = (((1-A)*self.solar_exposition)/(4*c))**(1/4)

            return kelvin
        elif self.type == "Gaz Giant":
            A = random.uniform(0.15,0.45)
            c = 5.67 * (10)**-8
            kelvin = (((1-A)*self.solar_exposition)/(4*c))**(1/4)
            
            return kelvin
    
## ----------------------------------------------------------------------------------------------------------------------
## Other function at the planet level
## ----------------------------------------------------------------------------------------------------------------------

    def initialisation_construction_factor (self):

        #definition of the penality due to gravity:
        k = 0.025
        self.facteur_construction_gravity = k * self.surface_gravity
        
        #definition of the penality due to temperature:

    
#################################################################################################################################
## ------------------------------------------------------------------------------------------------------------------------------
## Star class
## ------------------------------------------------------------------------------------------------------------------------------
#################################################################################################################################

class Stars():

    def __init__(self,x,y,type= None):
        self.type = type if type is not None else self.random_type()


        self.category = "Star"
        self.positionX = x
        self.positionY = y

        self.kelvin = None
        self.luminosity = None
        self.radius = None
        self.random_star_temperature_luminosity()
        self.color = self.compute_star_color()
        self.radius_pixel = 50

        print(self.kelvin)
        print(self.luminosity)
        print(self.radius_pixel)
        print(self.color)

## ----------------------------------------------------------------------------------------------------------------------
## Definition of deterministic characteristics 
## ----------------------------------------------------------------------------------------------------------------------

    def compute_star_color(self):
        """Return the RGB color corresponding to a given star type."""
        

        if self.type == "O":
            return (155, 176, 255)  # Blue
        elif self.type == "B":
            return (170, 191, 255)  # Blue-white
        elif self.type == "A":
            return (202, 215, 255)  # White
        elif self.type == "F":
            return (248, 247, 255)  # Yellow-white
        elif self.type == "G":
            return (255, 244, 234)  # Yellow
        elif self.type == "K":
            return (255, 210, 161)  # Orange
        elif self.type == "M":
            return (255, 204, 111)  # Red-orange
        

## ----------------------------------------------------------------------------------------------------------------------
## Definition of random characteristics 
## ----------------------------------------------------------------------------------------------------------------------



    def random_star_temperature_luminosity(self):
        if self.type == "O":
            # 30 000 – 50 000 K, 30 000 – 1 000 000 L☉, 6 – 15 R☉
            self.kelvin = random.randint(300, 500) * 100
            self.luminosity = random.uniform(3e4, 1e6)
            self.radius = random.uniform(6, 15)

        elif self.type == "B":
            # 10 000 – 30 000 K, 25 – 30 000 L☉, 2 – 6 R☉
            self.kelvin = random.randint(100, 300) * 100
            self.luminosity = random.uniform(25, 3e4)
            self.radius = random.uniform(2, 6)

        elif self.type == "A":
            # 7 500 – 10 000 K, 5 – 25 L☉, 1.4 – 2.5 R☉
            self.kelvin = random.randint(75, 100) * 100
            self.luminosity = random.uniform(5, 25)
            self.radius = random.uniform(1.4, 2.5)

        elif self.type == "F":
            # 6 000 – 7 500 K, 1.5 – 5 L☉, 1.15 – 1.4 R☉
            self.kelvin = random.randint(60, 75) * 100
            self.luminosity = random.uniform(1.5, 5)
            self.radius = random.uniform(1.15, 1.4)

        elif self.type == "G":
            # 5 200 – 6 000 K, 0.6 – 1.5 L☉, 0.9 – 1.1 R☉
            self.kelvin = random.randint(52, 60) * 100
            self.luminosity = random.uniform(0.6, 1.5)
            self.radius = random.uniform(0.9, 1.1)

        elif self.type == "K":
            # 3 900 – 5 200 K, 0.08 – 0.6 L☉, 0.7 – 0.9 R☉
            self.kelvin = random.randint(39, 52) * 100
            self.luminosity = random.uniform(0.08, 0.6)
            self.radius = random.uniform(0.7, 0.9)

        elif self.type == "M":
            # 2 400 – 3 900 K, 0.0001 – 0.08 L☉, 0.1 – 0.7 R☉
            self.kelvin = random.randint(24, 39) * 100
            self.luminosity = random.uniform(0.0001, 0.08)
            self.radius = random.uniform(0.1, 0.7)
        

    def random_type(self):
        """Return a random stellar spectral type."""
        return random.choice(["O", "B", "A", "F", "G", "K", "M"])











##ideas for later development


class fleet:

    def __init__(self):
        pass
        
