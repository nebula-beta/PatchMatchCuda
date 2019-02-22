#include <iostream>

#include "GlobalState.h"
#include "PatchMatch.h"

#include <iostream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

#include <sys/stat.h>
#include <sys/types.h>


// Includes CUDA
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_texture_types.h>
#include <vector_types.h>

float Evaluate(cv::Mat standard, cv::Mat myMap) 
{
	double sum = standard.rows * standard.cols;
	int count = 0;
	int ans = 0;
	for (int i = 0; i < standard.rows; i ++) 
	{
		for (int j = 0; j < standard.cols; j ++) 
		{
			ans = standard.ptr<uchar>(i)[j] - myMap.ptr<float>(i)[j];
			//与原图灰度相差大于1可认为是bad pixels，因为增强对比度，所以disparity maps都乘以3显示
			if (ans > 3 || ans < -3) count ++;
		}
	}
	double result = (count + 0.0)/sum;
	std::cout << std::setiosflags(std::ios::fixed);
	std::cout << std::setprecision(2) << result * 100 << "\\%" << std::endl;
	return result * 100;
}

bool check_image(const cv::Mat &image, std::string name="Image")
{
	if(!image.data)
	{
		std::cerr <<name <<" data not loaded.\n";
		return false;
	}
	return true;
}


bool check_dimensions(const cv::Mat &img1, const cv::Mat &img2)
{
	if(img1.cols != img2.cols or img1.rows != img2.rows)
	{
		std::cerr << "Images' dimensions do not corresponds.";
		return false;
	}
	return true;
}


void addImageToTextureFloatColor(std::vector<cv::Mat>& imgs, cudaTextureObject_t texs[], cudaArray* cuArray[])
{

	for(int i = 0; i < imgs.size(); ++i)
	{
		int rows = imgs[i].rows;
		int cols = imgs[i].cols;
		cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();

		cudaMallocArray(&cuArray[i], &channelDesc, cols, rows);
		cudaMemcpy2DToArray(cuArray[i], 0, 0, imgs[i].ptr<float>(), imgs[i].step[0], cols * sizeof(float) * 4, rows, cudaMemcpyHostToDevice);

		struct cudaResourceDesc res_desc;
		memset(&res_desc, 0, sizeof(res_desc));
		res_desc.resType = cudaResourceTypeArray;
		res_desc.res.array.array = cuArray[i];

		struct cudaTextureDesc tex_desc;
        memset(&tex_desc, 0, sizeof(tex_desc));
        tex_desc.addressMode[0]   = cudaAddressModeWrap;
        tex_desc.addressMode[1]   = cudaAddressModeWrap;
        tex_desc.filterMode       = cudaFilterModeLinear;
        tex_desc.readMode         = cudaReadModeElementType;
        tex_desc.normalizedCoords = 0;

        cudaCreateTextureObject(&(texs[i]), &res_desc, &tex_desc, NULL);
	}
}

void addImageToTextureFloatGray(std::vector<cv::Mat>& imgs, cudaTextureObject_t texs[], cudaArray* cuArray[])
{
	for(int i = 0; i < imgs.size(); ++i)
	{
		int rows = imgs[i].rows;
		int cols = imgs[i].cols;
		cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);

		cudaMallocArray(&cuArray[i], &channelDesc, cols, rows);
		cudaMemcpy2DToArray(cuArray[i], 0, 0, imgs[i].ptr<float>(), imgs[i].step[0], cols * sizeof(float), rows, cudaMemcpyHostToDevice);

		struct cudaResourceDesc res_desc;
		memset(&res_desc, 0, sizeof(res_desc));
		res_desc.resType = cudaResourceTypeArray;
		res_desc.res.array.array = cuArray[i];

		struct cudaTextureDesc tex_desc;
        memset(&tex_desc, 0, sizeof(tex_desc));
        tex_desc.addressMode[0]   = cudaAddressModeWrap;
        tex_desc.addressMode[1]   = cudaAddressModeWrap;
        tex_desc.filterMode       = cudaFilterModeLinear;
        tex_desc.readMode         = cudaReadModeElementType;
        tex_desc.normalizedCoords = 0;

        cudaCreateTextureObject(&(texs[i]), &res_desc, &tex_desc, NULL);
	}
}




std::vector<std::string> name = {"Aloe", "Baby1", "Baby2", "Baby3", "Bowling1",
"Bowling2", "Cloth1", "Cloth2", "Cloth3", "Cloth4", "Flowerpots",
"Lampshade1", "Lampshade2", "Midd1", "Midd2", "Monopoly",
"Plastic", "Rocks1", "Rocks2", "Wood1", "Wood2"};
int main(int argc, char** argv)
{
	
//  参数
	const float alpha =  0.9f;
	const float gamma = 10.0f;
	const float tau_c = 10.0f;
	const float tau_g =  2.0f;
	

	mkdir("resultImages1", S_IRWXU);

	const time_t start = time(NULL);

	for (int i = 0; i < name.size(); i ++)
	{

		std::cout << name[i] << std::endl;
	//在目的文件夹中创建相应的文件夹，以便存入图片
		std::string str = "resultImages1/" + name[i];
		const char * dir = str.c_str();
		mkdir(dir, S_IRWXU);

		//读取images文件夹中的源图片
		cv::Mat img1 = cv::imread("dataset/" + name[i] + "/view1.png" , cv::IMREAD_COLOR );
		cv::Mat img2 = cv::imread("dataset/" + name[i] + "/view5.png" , cv::IMREAD_COLOR );

		if ( (!img1.data) || (!img2.data))

		{
			printf("Please input right data~~\n");
			return -1;
		}
		
		// Image loading check
		if(!check_image(img1, "Image 1") or !check_image(img2, "Image 2"))
			return 1;
		
		// Image sizes check
		if(!check_dimensions(img1, img2))
			return 1;
		
		
		AlgorithmParameters* params = new AlgorithmParameters();
		params->min_disparity = 1.0f;
		params->max_disparity = 80.f;
		params->box_width = 11;
		params->box_height = 11;
		params->tau_color = tau_c;
		params->tau_gradient = tau_g;
		params->alpha = alpha;
		params->gamma = gamma;


		GlobalState* gs = new GlobalState();
		gs->params = params;

		int rows = img1.rows;
		int cols = img1.cols;


		std::vector<cv::Mat> imgs = {img1, img2};

		if(params->color_processing)
		{
			std::vector<cv::Mat> img_color_float_alpha(2);
			std::vector<cv::Mat> img_color_float(2);
			for(int i = 0; i < imgs.size(); ++i)
			{
				img_color_float_alpha[i] = cv::Mat::zeros(rows, cols, CV_32FC4);
				cv::Mat alpha(rows, cols, CV_32FC1);

				std::vector<cv::Mat> channels(3);
				imgs[i].convertTo(img_color_float[i], CV_32FC3);

				cv::split(img_color_float[i], channels);
				channels.push_back(alpha);

				cv::merge(channels, img_color_float_alpha[i]);
			}
			addImageToTextureFloatColor(img_color_float_alpha, gs->imgs, gs->cuArray);
		}
		else
		{
			std::vector<cv::Mat> img_grayscale(2);
			std::vector<cv::Mat> img_grayscale_float(2);
			for(int i = 0; i < imgs.size(); ++i)
			{
				cv::cvtColor(imgs[i], img_grayscale[i], cv::COLOR_BGR2GRAY);
				img_grayscale[i].convertTo(img_grayscale_float[i], CV_32F, 1.0 / 255);
			}
			addImageToTextureFloatGray(img_grayscale_float, gs->imgs, gs->cuArray);
		}

		gs->lines->resize(rows, cols);

		runCuda(*gs);	
		cudaDeviceSynchronize();


		cv::Mat disp(rows, cols, CV_32FC1);

		for(int i = 0; i < rows; ++i)
		{
			for(int j = 0; j < cols; ++j)
			{
				disp.at<float>(i, j) = gs->lines->norm4[cols * i + j].w;
			}
		}


		for(int i = 0; i < 2; ++i)
		{
			cudaFreeArray(gs->cuArray[i]);
			cudaDestroyTextureObject(gs->imgs[i]);
		}
		delete gs;
		delete params;
		cudaDeviceSynchronize();


		try
		{

			cv::medianBlur(disp, disp, 3);
			disp = disp * 3;
			cv::imwrite( "resultImages1/" + name[i] + "/" + name[i] + "_disp1.png", disp);

			cv::Mat standardLeft = cv::imread("dataset/" + name[i] + "/disp1.png", -1);
			float error_rate_left = Evaluate(standardLeft, disp);

			disp.convertTo(disp, CV_8U);
			cv::imshow("disp", disp);
			cv::waitKey(1);

		} 
		catch(std::exception &e)
		{
			std::cerr << "Disparity save error.\n" <<e.what();
			return 1;
		}

		checkForLastCudaError();


	} 


	


	return 0;
}
