import pygame

def toggle_fullscreen(current_fullscreen_status):
    return not current_fullscreen_status

def event_check(events, fullscreen, state):
    width, height = state.image_size
    aspect_ratio_local = width / height
    for event in events:
        if event.type == pygame.QUIT:
            state.run_mode = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q:
                state.run_mode = False
                pygame.quit()
            elif event.key == pygame.K_f:
                fullscreen = toggle_fullscreen(fullscreen)
                state.needs_update = True
            elif event.key == pygame.K_r:
                # Update rotation immediately.
                state.rotation = (state.rotation + 45) % 360
                state.needs_update = True
            elif event.key == pygame.K_m:
                state.mirror = 1 - state.mirror
                state.needs_update = True
        elif event.type == pygame.VIDEORESIZE:
            new_width, new_height = event.size
            if new_width / new_height > aspect_ratio_local:
                new_width = int(new_height * aspect_ratio_local)
            else:
                new_height = int(new_width / aspect_ratio_local)
            state.image_size = (new_width, new_height)
            state.needs_update = True
    return fullscreen
