import pygame


def draw_circle (screen,color,x,y,radius):
    pygame.draw.circle(screen,color,(x,y), radius)


def draw_rectangle(
    screen, color, x, y, width, height, *,
    border_thickness=0, border_color=None,
    border_top=True, border_right=True,
    border_bottom=True, border_left=True
):
    # fill interior
    if color is not None:
        pygame.draw.rect(screen, color, (x, y, width, height))

    # no border = done
    if border_thickness <= 0:
        return

    # default border color
    if border_color is None:
        border_color = (0, 0, 0)

    t = border_thickness

    # Draw borders OUTSIDE the rectangle using lines
    if border_top: pygame.draw.line(screen, border_color, (x, y - t//2), (x + width, y - t//2), t)

    if border_bottom: pygame.draw.line(screen, border_color, (x, y + height + t//2), (x + width, y + height + t//2), t)

    if border_left: pygame.draw.line(screen, border_color, (x - t//2, y), (x - t//2, y + height), t)

    if border_right: pygame.draw.line(screen, border_color, (x + width + t//2, y), (x + width + t//2, y + height), t)




    
def draw_text (screen,x,y,text,size,color,given_font=None):
    font = pygame.font.Font(given_font, size)
    text_surface = font.render(text, True, color)
    screen.blit(text_surface, (x, y)) 