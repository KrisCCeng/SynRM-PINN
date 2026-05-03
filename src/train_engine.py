import torch

def train_one_epoch(model, dataloader, optimizer, loss_weighter, criterion, device):
    model.train()
    total_loss, total_mse = 0.0, 0.0
    total_pde1, total_pde2, total_sym = 0.0, 0.0, 0.0

    for batch_idx, (inputs, targets, id_iq_now, id_iq_next, we) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)
        id_iq_now, id_iq_next = id_iq_now.to(device), id_iq_next.to(device)
        we = we.to(device)

        optimizer.zero_grad()

        # [Modification]: ANN and SCN do not use physical loss functions; they rely purely on MSE backpropagation.
        if model.mode in ["ANN", "SCN"]:
            flux_pred = model(inputs, id_iq_now, id_iq_next, we)
            loss = criterion(flux_pred, targets)
            mse_val = loss.item()
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        # PINN enables physical losses and PCGrad surgery.
        elif model.mode == "PINN":
            flux_pred, loss_pde1, loss_pde2, loss_sym = model(inputs, id_iq_now, id_iq_next, we)
            mse_loss = criterion(flux_pred, targets)
            mse_val = mse_loss.item()
            
            weighted_losses = loss_weighter([mse_loss, loss_pde1, loss_pde2, loss_sym])
            
            total_pde1 += loss_pde1.item()
            total_pde2 += loss_pde2.item()
            total_sym += loss_sym.item()

            optimizer.pc_backward(weighted_losses)
            optimizer.step()
            
            total_loss += sum(l.item() for l in weighted_losses)

        total_mse += mse_val

    avg_total_loss = total_loss / len(dataloader)
    avg_mse_loss = total_mse / len(dataloader)

    weights_dict = {}
    if model.mode == "PINN" and loss_weighter is not None:
        avg_pde1 = total_pde1 / len(dataloader)
        avg_pde2 = total_pde2 / len(dataloader)
        avg_sym = total_sym / len(dataloader)
        
        with torch.no_grad():
            ws = loss_weighter.get_weights() 
            weights_dict = {
                'w_mse': float(ws[0]),
                'w_pde1': float(ws[1]),
                'w_pde2': float(ws[2]),
                'w_sym': float(ws[3]),
                'loss_mse': avg_mse_loss,
                'loss_pde1': avg_pde1,
                'loss_pde2': avg_pde2,
                'loss_sym': avg_sym
            }

    return avg_total_loss, avg_mse_loss, weights_dict


def validate(model, dataloader, criterion, device):
    """Validation function"""
    model.eval()
    total_mse = 0.0
    with torch.no_grad():
        for inputs, targets, id_iq_now, id_iq_next, we in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            id_iq_now = id_iq_now.to(device)
            we = we.to(device)
            
            if model.mode in ["PINN", "SCN"]:
                flux_pred = model.forward_flux(id_iq_now)
            else: # ANN
                flux_pred = model.forward_flux(id_iq_now)
                
            loss = criterion(flux_pred, targets)
            total_mse += loss.item()
            
    return total_mse / len(dataloader)