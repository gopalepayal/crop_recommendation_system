from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from django.core.paginator import Paginator
import json
import csv
import io
from . import utils


# ─────────────────────────────────────────────
#  Existing views (unchanged functionality)
# ─────────────────────────────────────────────

def landing(request):
    stats     = utils.get_overview_stats()
    best_info = utils.get_best_model_info()
    context = {
        'stats':       stats,
        'best_info':   best_info,
        'active_page': 'landing',
    }
    return render(request, 'landing.html', context)


def overview(request):
    stats      = utils.get_overview_stats()
    crop_chart = utils.generate_crop_distribution_chart()
    context = {
        'stats':       stats,
        'crop_chart':  crop_chart,
        'active_page': 'overview',
    }
    return render(request, 'dashboard.html', context)


def crop_analysis(request):
    nutrient_charts = utils.generate_nutrient_charts()
    env_charts      = utils.generate_environmental_charts_crop()
    profile_data    = utils.get_crop_profile_data()

    # If a crop is requested via GET param, render comparison charts for it
    selected_crop = request.GET.get('crop', '').strip().lower()
    comparison_charts = None
    selected_profile = None
    if selected_crop and selected_crop in profile_data['profiles']:
        comparison_charts = utils.generate_crop_profile_comparison_charts(selected_crop)
        selected_profile  = profile_data['profiles'][selected_crop]
        selected_profile['name'] = selected_crop

    context = {
        'nutrient_charts':   nutrient_charts,
        'env_charts':        env_charts,
        'crop_list':         profile_data['crop_list'],
        'profiles_json':     json.dumps(profile_data['profiles']),
        'overall_avgs':      profile_data['overall_avgs'],
        'selected_crop':     selected_crop,
        'selected_profile':  selected_profile,
        'comparison_charts': comparison_charts,
        'active_page':       'crop_analysis',
    }
    return render(request, 'crop_analysis.html', context)


def environmental_analysis(request):
    charts = utils.generate_environmental_analysis_charts()
    context = {
        'charts':      charts,
        'active_page': 'environmental_analysis',
    }
    return render(request, 'environmental_analysis.html', context)


def model_performance(request):
    results_df  = utils.load_results()
    results_records = results_df.to_dict('records')
    best_info   = utils.get_best_model_info()
    charts      = utils.generate_model_performance_charts()
    fi_chart    = utils.generate_feature_importance_chart()
    fi_table    = utils.get_feature_importance_table_data()
    cm_chart    = utils.generate_confusion_matrix_chart()

    context = {
        'results':     results_records,
        'best_info':   best_info,
        'charts':      charts,
        'fi_chart':    fi_chart,
        'fi_table':    fi_table,
        'cm_chart':    cm_chart,
        'active_page': 'model_performance',
    }
    return render(request, 'model_performance.html', context)


def crop_recommendation(request):
    prediction     = None
    raw_pred       = None
    confidence_pct = None
    top3           = None
    error          = None
    crop_avgs      = None
    user_input     = None
    why_chart      = None
    why_comparison = None

    if request.method == 'POST':
        try:
            n_raw        = request.POST.get('n', '').strip()
            p_raw        = request.POST.get('p', '').strip()
            k_raw        = request.POST.get('k', '').strip()
            temp_raw     = request.POST.get('temp', '').strip()
            humidity_raw = request.POST.get('humidity', '').strip()
            ph_raw       = request.POST.get('ph', '').strip()
            rainfall_raw = request.POST.get('rainfall', '').strip()

            if not all([n_raw, p_raw, k_raw, temp_raw, humidity_raw, ph_raw, rainfall_raw]):
                error = 'All seven input fields are required.'
            else:
                n        = float(n_raw)
                p        = float(p_raw)
                k        = float(k_raw)
                temp     = float(temp_raw)
                humidity = float(humidity_raw)
                ph       = float(ph_raw)
                rainfall = float(rainfall_raw)

                if n < 0 or n > 300:
                    error = 'Nitrogen (N) must be a non-negative number between 0 and 300.'
                elif p < 0 or p > 300:
                    error = 'Phosphorus (P) must be a non-negative number between 0 and 300.'
                elif k < 0 or k > 300:
                    error = 'Potassium (K) must be a non-negative number between 0 and 300.'
                elif temp < -10 or temp > 60:
                    error = 'Temperature must be between -10°C and 60°C.'
                elif humidity < 0 or humidity > 100:
                    error = 'Humidity must be between 0% and 100%.'
                elif ph < 0 or ph > 14:
                    error = 'Soil pH must be between 0 and 14.'
                elif rainfall < 0:
                    error = 'Rainfall must be a non-negative number (0 mm or greater).'
                else:
                    res            = utils.predict_crop_with_probabilities(n, p, k, temp, humidity, ph, rainfall)
                    prediction     = res['prediction']
                    raw_pred       = res['raw_pred']
                    confidence_pct = res['confidence_pct']
                    top3           = res['top3']

                    user_input = {
                        'N': n, 'P': p, 'K': k,
                        'temperature': temp, 'humidity': humidity,
                        'ph': ph, 'rainfall': rainfall,
                    }
                    crop_avgs      = utils.get_crop_averages(raw_pred)
                    why_comparison = utils.get_why_comparison_data(user_input, raw_pred)
                    why_chart      = utils.generate_why_chart(user_input, raw_pred)

        except (ValueError, TypeError):
            error = 'Please provide valid numeric inputs for all fields.'
        except Exception as e:
            error = f'An error occurred during prediction: {str(e)}'

    context = {
        'prediction':     prediction,
        'raw_pred':       raw_pred,
        'confidence_pct': confidence_pct,
        'top3':           top3,
        'error':          error,
        'stats':          utils.get_overview_stats(),
        'user_input':     user_input,
        'crop_avgs':      crop_avgs,
        'why_comparison': why_comparison,
        'why_chart':      why_chart,
        'active_page':    'crop_recommendation',
    }
    return render(request, 'recommendation.html', context)


# ─────────────────────────────────────────────
#  Dataset Explorer (new view)
# ─────────────────────────────────────────────

def dataset_explorer(request):
    df = utils.load_data()

    # Filtering
    crop_filter  = request.GET.get('crop', '').strip().lower()
    search_query = request.GET.get('search', '').strip().lower()

    filtered_df = df.copy()
    if crop_filter:
        filtered_df = filtered_df[filtered_df['label'].str.lower() == crop_filter]
    if search_query:
        filtered_df = filtered_df[filtered_df['label'].str.lower().str.contains(search_query)]

    total_filtered = len(filtered_df)
    crop_list = sorted(df['label'].unique().tolist())

    # CSV download
    if request.GET.get('download') == '1':
        response = HttpResponse(content_type='text/csv')
        response['Content-Disposition'] = 'attachment; filename="crop_recommendation_filtered.csv"'
        writer = csv.writer(response)
        writer.writerow(['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall', 'label'])
        for _, row in filtered_df.iterrows():
            writer.writerow([
                row['N'], row['P'], row['K'],
                row['temperature'], row['humidity'], row['ph'], row['rainfall'],
                row['label'],
            ])
        return response

    # Pagination
    records = filtered_df.round(3).to_dict('records')
    paginator = Paginator(records, 25)
    page_number = request.GET.get('page', 1)
    page_obj = paginator.get_page(page_number)

    context = {
        'page_obj':      page_obj,
        'total_filtered': total_filtered,
        'total_records':  len(df),
        'crop_list':      crop_list,
        'crop_filter':    crop_filter,
        'search_query':   search_query,
        'active_page':    'dataset_explorer',
    }
    return render(request, 'dataset_explorer.html', context)
