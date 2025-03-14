def toggle_fullscreen(current_fullscreen_status):
    # Lazy import to avoid circular dependency.
    import image_display
    new_fullscreen = not current_fullscreen_status
    image_display.display_init(new_fullscreen)
    return new_fullscreen

def event_check(fullscreen):
    import pygame
    import image_display
    width, height = image_display.image_size
    aspect_ratio_local = width / height
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            image_display.run_mode = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q:
                image_display.run_mode = False
                pygame.quit()
            if event.key == pygame.K_f:
                fullscreen = toggle_fullscreen(fullscreen)
            if event.key == pygame.K_r:
                image_display.rotation_angle = (image_display.rotation_angle + 45) % 360
                image_display.display_init(fullscreen)
            if event.key == pygame.K_m:
                image_display.mirror_mode = 1 - image_display.mirror_mode
                image_display.display_init(fullscreen)
        elif event.type == pygame.VIDEORESIZE:
            new_width, new_height = event.size
            if new_width / new_height > aspect_ratio_local:
                new_width = int(new_height * aspect_ratio_local)
                image_display.image_size = (new_width, new_height)
            else:
                new_height = int(new_width / aspect_ratio_local)
                image_display.image_size = (new_width, new_height)
            image_display.display_init(fullscreen)
    return fullscreen
