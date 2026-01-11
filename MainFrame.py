# Extensions
import pygame
import sys

#Import of other pages
import Solar_System
import camera
import UI_manager 


pygame.init()


width = 1280
height = 720

zoom_level = 1

screen = pygame.display.set_mode((width,height))
pygame.display.set_caption("Orion Game")



Solar_System.create_first_system (screen)
current_system = Solar_System.Solar_system_list[0]

camera = camera.camera(screen)



dragging = False
last_mouse = (0, 0)
running = True

UI_manager.solar_map_ui_configuration (width,height)

while running:
    
    screen.fill((0, 0, 0))  # Fond noir
    #current_system.System_rendering(screen)
    camera.System_rendering(screen,zoom_level,current_system)
    camera.Ui_rendering(screen)
    pygame.display.flip()  # Met à jour l'écran

    




    # Gestion des événements
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        
        if event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                x = event.pos[0]
                y = event.pos[1]


                debug_clicked_object = camera.screen_got_clicked(x,y,current_system)
                print(f"Debug clicked object: {debug_clicked_object}")

        if event.type == pygame.MOUSEWHEEL:
            camera.zoom(event.y)


        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            dragging = True
            last_mouse = event.pos

        if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            dragging = False

        if event.type == pygame.MOUSEMOTION and dragging:
            mx, my = event.pos
            dx = mx - last_mouse[0]
            dy = my - last_mouse[1]

            camera.move_camera(dx, dy)
            last_mouse = (mx, my)

# Fermer correctement
pygame.quit()
sys.exit()