
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch_harmonics.examples import ShallowWaterSolver
import torch_harmonics as harmonics
from torch.profiler import profile, ProfilerActivity

from tqdm import tqdm

import matplotlib.pyplot as plt



device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def l2loss_sphere(solver, prd, tar, relative=True, squared=False):
    """
    Loss function is already implemented
    """
    loss = solver.integrate_grid((prd - tar)**2, dimensionless=True).sum(dim=-1)
    if relative:
        loss = loss / solver.integrate_grid(tar**2, dimensionless=True).sum(dim=-1)
    
    if not squared:
        loss = torch.sqrt(loss)
    loss = loss.mean()

    return loss


def train(model_holder, train_loader, valid_loader, optimizer, device, num_epochs,current_epoch,save_path):
    print('Device being used: ',torch.cuda.get_device_name(device))
    print('Memory before model to device ',torch.cuda.memory_allocated()/(1024**2))
    model_holder.model.to(device)  # Move model to device (CPU or GPU)
    solver = ShallowWaterSolver(model_holder.model.img_size[0],model_holder.model.img_size[1],dt=0, grid = model_holder.model.grid).to(device)
    print('Memory before epoch start ',torch.cuda.memory_allocated()/(1024**2))


    
    for epoch in range(current_epoch,num_epochs):
        model_holder.model.train()  # Set model to training mode
        optimizer.zero_grad()
        epoch_loss = 0.0  # Track loss for the epoch
        
        for batch_idx, (inputs, targets) in enumerate(tqdm(train_loader)):
            # Move inputs and targets to the device
            #print('Data batch in')
            inputs, targets = inputs.to(device), targets.to(device)
           
            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model_holder.model(inputs)
            # Compute loss
            loss = l2loss_sphere(solver,outputs, targets)

            # Backward pass
            loss.backward()

            # Update parameters
            optimizer.step()

            # Accumulate epoch loss
            epoch_loss += loss.item()
            #del loss
            del outputs
        
        model_holder.model.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(tqdm(valid_loader,leave=False)):
                inputs, targets = inputs.to(device), targets.to(device)
                # Forward pass
                outputs = model_holder.model(inputs)
                # Compute loss
                loss = l2loss_sphere(solver,outputs, targets)

                valid_loss += loss.item()
        ave_val_loss = valid_loss/len(valid_loader)
        if ave_val_loss<model_holder.current_valid_ave:
            model_holder.set_ave_valid(ave_val_loss)
            model_holder.save_model(epoch+1,optimizer,loss,save_path)
        # Print epoch metrics

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss / len(train_loader):.4f}")



def train_profiling(model_holder, train_loader, optimizer, device, num_epochs,current_epoch,save_path):
    """
    Args:
        model (torch.nn.Module): The model to train.
        train_loader (DataLoader): DataLoader for the training dataset.
        criterion (nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer for model parameters.
        device (torch.device): Device to run the training on (e.g., 'cuda' or 'cpu').
        num_epochs (int): Number of epochs to train for.
    """
    print('Memory before model to device ',torch.cuda.memory_allocated()/(1024**2))
    model_holder.model.to(device)  # Move model to device (CPU or GPU)
    model_holder.model.train()  # Set model to training mode
    solver = ShallowWaterSolver(model_holder.model.img_size[0],model_holder.model.img_size[1],dt=0, grid = model_holder.model.grid).to(device)
    print('Memory before epoch start ',torch.cuda.memory_allocated()/(1024**2))
    with profile(
    activities=[ProfilerActivity.CUDA],  # Profile both CPU and GPU  
    record_shapes=True,  # Record tensor shapes
    profile_memory=True,  # Track memory usage
    with_stack=True  # Include stack traces
) as prof:
        for epoch in range(current_epoch,num_epochs):
            optimizer.zero_grad()
            epoch_loss = 0.0  # Track loss for the epoch
            
            for batch_idx, (inputs, targets) in enumerate(tqdm(train_loader,leave=False)):
                # Move inputs and targets to the device
                inputs, targets = inputs.to(device), targets.to(device)
                #print('Memory after databatch to gpu',torch.cuda.memory_allocated()/(1024**2))
                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                outputs = model_holder.model(inputs)

                # Compute loss
                loss = l2loss_sphere(solver,outputs, targets)

                # Backward pass
                loss.backward()

                # Update parameters
                optimizer.step()

                # Accumulate epoch loss
                epoch_loss += loss.item()
                #del loss
                del outputs
            

            model_holder.save_model(epoch,optimizer,loss,save_path)
            # Print epoch metrics

            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss / len(train_loader):.4f}")
    print(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=10))


def evaluate_year(model_holder, test_loader,outcomes_holder,loss_holder, device):
    """
    Evaluates the model on the provided data.

    Args:
        model (torch.nn.Module): The trained PyTorch model to evaluate.
        dataloader (torch.utils.data.DataLoader): DataLoader for the evaluation dataset.
        criterion (torch.nn.Module, optional): Loss function to compute loss, if needed.
        device (str): Device to run the evaluation on ('cpu' or 'cuda').

    Returns:
        dict: A dictionary containing evaluation metrics such as loss and accuracy.
    """
    
    total_loss = 0.0
    solver = ShallowWaterSolver(model_holder.model.img_size[0],model_holder.model.img_size[1],dt=0, grid = model_holder.model.grid).to(device)

    model_holder.model.to(device)

    # tot_length = len(test_loader) + 6

    # output_length = tot_length//6



    # outcomes_holder = torch.empty((output_length,n_channels,lat,lon),dtype=torch.float32,device=device)
    # loss_holder = torch.empty(output_length,dtype=torch.float32,device=device)

    # outcomes_holder.to(device)
    # loss_holder.to(device)
    print('Output size = ',len(outcomes_holder))
    model_holder.model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # Disable gradient computation

        for idx,(inputs,targets) in enumerate(tqdm(test_loader)):
            inputs, targets = inputs.to(device), targets.to(device)
            output_index = idx//6
            if idx//6 >= len(outcomes_holder):
                break
            elif idx==0:
                # print('inputs ',inputs.is_cuda)
                outputs = model_holder.model(inputs)  # Forward pass
                outputs.to(device)
            elif idx%6==0:
                # print('whole outcomes ',outcomes_holder.is_cuda)
                # print('outcomes idx',outcomes_holder[idx-1].is_cuda)
                # print('shapes ',outcomes_holder[idx-1].shape)
                outputs = model_holder.model(outcomes_holder[output_index-1:output_index,:,:,:])
                outputs.to(device)
            else:
                continue
            outcomes_holder[output_index,:,:,:] = outputs.unsqueeze(0)

            loss = l2loss_sphere(solver,outputs,targets)

            loss_value = loss.item()
            
            loss_holder[output_index] = loss_value

    # Calculate average loss and accuracy
    
    return loss_holder,outcomes_holder


def format_outputs(model_holder,grid='equiangular'):
    import cartopy.crs as ccrs

    nlat = model_holder.model.img_size[0]
    nlon = model_holder.model.img_size[1]
    if grid == 'equiangular':
        cost, quad_weights = harmonics.quadrature.clenshaw_curtiss_weights(nlat, -1, 1)

    quad_weights = torch.as_tensor(quad_weights).reshape(-1, 1)

    # apply cosine transform and flip them
    lats = -torch.as_tensor(np.arcsin(cost))
    lons = torch.linspace(0, 2*np.pi, nlon+1, dtype=torch.float64)[:nlon]

    lons = lons.squeeze() - torch.pi
    lats = lats.squeeze()

    Lons, Lats = np.meshgrid(lons, lats)
    
    proj = ccrs.Orthographic(central_longitude=0.0, central_latitude=25.0)

    Lons = Lons*180/np.pi
    Lats = Lats*180/np.pi

    return proj,Lats,Lons

def plot_data(fig, proj,Lats,Lons,data, cmap='twilight_shifted', vmax=None, vmin=None,antialiased=False,title='None'):
    import cartopy.crs as ccrs
    ax = fig.add_subplot(projection=proj)
    im = ax.pcolormesh(Lons, Lats, data, cmap=cmap, transform=ccrs.PlateCarree(),antialiased=antialiased, vmax=vmax, vmin=vmin)
    plt.title(title, y=1.05)

    return im