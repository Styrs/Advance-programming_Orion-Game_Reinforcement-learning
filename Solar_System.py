import objects




#__This list contain an object for every star system
Solar_system_list = []



def create_first_system (screen):
        name = "Orion"
        maxSize = 8
        planetnumber = 3
        bodynumber = 3
        StarType = "F"

        system_preconfiguration = { #Work by adding the parameters you want for planet. IN THE RIGHT ORDER 
                                    #type = None,name = None,masse = None,radius = None,surface_kelvin =None
            "planets": [
                {"type": "Telluric", "name": "Eve", "masse": 1, "radius": 1, "surface_kelvin": 300},
                {"type": "Telluric", "name": "Orion", "masse": 0.8, "radius": 0.9, "surface_kelvin": 200},
                {"type": None, "name": None, "masse": None, "radius": None, "surface_kelvin": None}
            ]
        }

        first_system = Solar_system(name,maxSize,planetnumber,bodynumber,StarType,screen,system_preconfiguration)
        Solar_system_list.append(first_system)
        
        return first_system




#################################################################################################################################
## ------------------------------------------------------------------------------------------------------------------------------
## Solar_system class
## ------------------------------------------------------------------------------------------------------------------------------
#################################################################################################################################




class Solar_system:
    number_system = 1



    def __init__(self,name,maxsize,planetnumber,bodynumber,StarType,screen,system_preconfiguration = None):


        center_x = screen.get_width() // 2
        center_y = screen.get_height() // 2

        self.id = Solar_system.number_system
        self.name = name
        self.maxSize = maxsize
        self.pixel_UA = 75 #default value
        self.Screen_scale_rendering(screen.get_width(),screen.get_height())
        
        self.system_preconfiguration = system_preconfiguration
        self.planetnumber = planetnumber
        self.bodynumber = bodynumber
        self.StarType = StarType
        self.planet_list = []
        self.star_list = []
        self.entity_list = [self.star_list,self.planet_list] #list of all entities in the system (planets, moons, asteroids, comets, etc...)
        
        self.Star_generation(center_x,center_y)


        if self.planetnumber > 0:
             self.Planet_generation(center_x,center_y)

        Solar_system.number_system = Solar_system.number_system + 1

## ------------------------------------------------------------------------------------------------------------------------------
## configurations functions of the solar system
## ------------------------------------------------------------------------------------------------------------------------------


    def Planet_generation (self,center_x,center_y):
        counter_planet = 0
        planets_preconfiguration = self.system_preconfiguration.get("planets")

        if planets_preconfiguration != None:
            

            for one_planets_preconfiguration in planets_preconfiguration:
                counter_planet = counter_planet + 1
                distance = self.maxSize /self.planetnumber * counter_planet
                x = center_x + int(distance * self.pixel_UA)

                planet = objects.planets_Moon(
                    distance,
                    self.star_list[0],
                    x,
                    center_y,
                    type=one_planets_preconfiguration.get("type"),
                    name=one_planets_preconfiguration.get("name"),
                    masse=one_planets_preconfiguration.get("masse"),
                    radius=one_planets_preconfiguration.get("radius"),
                    surface_kelvin=one_planets_preconfiguration.get("surface_kelvin"),
                )

                self.planet_list.append(planet)

        else :
            for n in range(self.planetnumber):
                counter_planet = counter_planet + 1
                distance = self.maxSize /self.planetnumber * counter_planet
                x = center_x + int(distance * self.pixel_UA)
                planet = objects.planets_Moon (distance,self.star_list[0],x,center_y)

                self.planet_list.append(planet)
        

    def Star_generation (self,center_x,center_y):

        star = objects.Stars(center_x,center_y,self.StarType)
        self.star_list.append(star)


    def Screen_scale_rendering(self,screen_width,screen_height):
        lowest_side = min(screen_width,screen_height) 
        self.pixel_UA = lowest_side / (self.maxSize * 1.1 * 2) #2 bc maxsize is the radius and 1.1 to create a bit of space at the end of the screen
    
## ------------------------------------------------------------------------------------------------------------------------------
## Screen interaction functions
## ------------------------------------------------------------------------------------------------------------------------------

    def get_entities_from_coordinates (self,click_x,click_y):
        candidates = []
        
        for entity_list in self.entity_list:
            for entity in entity_list:
                x = entity.positionX
                y = entity.positionY
                radius = entity.radius_pixel

                dx = click_x - x
                dy = click_y - y
                distance = (dx**2 + dy**2)**0.5
                
                if distance <= radius:
                    candidates.append((distance, entity))
                    if entity.category == "Planetoid" or entity.category == "Star":
                        return entity
                    
        if len(candidates) > 0:
            return candidates[0][1]  # Return the fisrt entity
            
                    
        return None
    

## ------------------------------------------------------------------------------------------------------------------------------
## get object information functions
## ------------------------------------------------------------------------------------------------------------------------------